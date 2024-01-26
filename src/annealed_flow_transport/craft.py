"""Implementation of the Continual Repeated Annealed Flow Transport (CRAFT) sampler algorithm."""

import jax
import jax.numpy as jnp
import optax
from absl import logging
from time import time
import chex
from typing import Tuple, Callable
from .utils.aft_types import InitialDensitySampler, LogDensityByTemp
from .utils.hmc import HMCKernel
from .utils.smc_utils import update_step_with_flow, estimate_free_energy

def craft_step(key: jax.Array, samples: jax.Array, log_weights: jax.Array, 
               flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]], 
               flow_params: dict, kernel: HMCKernel, log_density: LogDensityByTemp, 
               beta: float, beta_prev: float, step: int, threshold: float, 
               loss_val_and_grad: Callable[[jax.Array, jax.Array, dict, float, float], 
                                            Tuple[float, jax.Array]], 
               ) -> Tuple[jax.Array, jax.Array, float, float, float, jax.Array]:
  """A temperature step of CRAFT that updates the input samples and their 
  log weights.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  samples : jax.Array
    An array containing the batch of particles.
  log_weights : jax.Array
    An array containing the log weights of the particles in samples.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  kernel : HMCKernel
    The HMC Kernel to be applied in the MCMC step.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  beta : float
    The temperature at the current iteration.
  beta_prev : float
    The temperature at the previous iteration.
  step : int
    The current iteration time step, to input into kernel in case it uses 
    a StepSizeSchedule.
  threshold : float
    The ESS threshold to trigger resampling.
  loss_val_and_grad : Callable[[jax.Array, jax.Array, dict, float, float], 
                               Tuple[float, jax.Array]]
    A function that takes in an array of samples, their log weights, 
    parameters of the flow, and the current and previous annealing 
    temperatures, and returns the value and gradient of the 
    variational free energy.
  
  Returns
  -------
  new_samples : jax.Array
    The array containing the batch of updated particles.
  new_log_weights : jax.Array
    The array containing the updated log weights of the 
    particles in new_samples.
  log_evidence_increment : float
    The estimate of log Z_t - log Z_{t-1}, where Z_t is the normalizing 
    constant of the t'th bridging distribution.
  acpt_rate : float
    Average acceptance rate of all HMC moves in this batch of particles.
  vfe : float
    The variational free energy, or loss, of the flow model.
  vfe_grad : jax.Array
    The gradient of the variational free energy, or loss, of 
    the flow model with respect to flow_params.
  """
  vfe, vfe_grad = loss_val_and_grad(samples, log_weights, flow_params, 
                                    beta, beta_prev)
  new_samples, new_log_weights, \
    log_evidence_increment, acpt_rate = update_step_with_flow(key, samples, 
                                                              log_weights, 
                                                              flow_apply, 
                                                              flow_params, 
                                                              log_density, 
                                                              beta, beta_prev, 
                                                              kernel, threshold, 
                                                              step)
  
  return new_samples, new_log_weights, log_evidence_increment, acpt_rate, vfe, vfe_grad

def craft_loop(key: jax.Array, sampler: InitialDensitySampler, 
               flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]], 
               transition_params: dict, opt: optax.GradientTransformation, 
               opt_states: optax.OptState, kernel: HMCKernel, 
               log_density: LogDensityByTemp, threshold: float, num_temps: int, 
               loss_val_and_grad: Callable[[jax.Array, jax.Array, dict, float, float], 
                                            Tuple[float, jax.Array]], 
               betas: jax.Array | None = None) -> Tuple[jax.Array, jax.Array, jax.Array, 
                                                        dict, optax.OptState, float, float]:
  """A single run of the CRAFT algorithm, with flow parameter updates.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  sampler : InitialDensitySampler
    A callable that takes a jax PRNG key and returns an array 
    of particles following the initial distribution.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model.
  transition_params : dict
    `num_temps` flow parameters stacked together, to scan through 
    and pushed forward through the flow model in flow_apply.
  opt : optax.GradientTransformation
    The optimizer of the flow model.
  opt_states : optax.OptState
    `num_temps` optimization states stacked together, to record the 
    optimization states of each of the flow parameters in transition_params.
  kernel : HMCKernel
    The HMC kernel to be applied in each mutation step.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  threshold : float
    The ESS threshold to trigger resampling.
  num_temps : int
    The number of annealing temperatures to be used.
  loss_val_and_grad : Callable[[jax.Array, jax.Array, dict, float, float], 
                               Tuple[float, jax.Array]]
    A function that takes in an array of samples, their log weights, 
    parameters of the flow, and the current and previous annealing 
    temperatures, and returns the value and gradient of the 
    variational free energy.
  betas : jax.Array | None = None
    An optional argument for the array of temperatures to be used by 
    the CRAFT algorithm. If not specified, defaults to a geometric 
    annealing schedule.

  Returns
  -------
  final_samples : jax.Array
    The array of particles produced after the CRAFT algorithm.
  final_log_weights : jax.Array
    An array containing the log weights of the particles in final_samples.
  acpt_rate : jax.Array
    An array containing the acceptance rates of the HMC kernel 
    at each annealing temperature.
  final_transition_params : dict
    The updated flow parameters after learning from this CRAFT iteration.
  final_opt_states : optax.OptState
    The updated optimization states of the transition_params.
  log_evidence_estimate : float
    The log evidence estimate produced from this run of CRAFT (with 
    pre-update transition_params used for the flows.)
  total_vfe : float
    The total loss of the flow models from this run of CRAFT.
  """
  key, key_ = jax.random.split(key)
  initial_samples = sampler(key_)
  initial_log_weights = -jnp.log(sampler.num_particles) * jnp.ones(sampler.num_particles)

  def scan_step(state, per_step_input):
    samples, log_weights, beta_prev = state
    key, params, beta, step = per_step_input
    new_samples, new_log_weights, \
    log_evidence_increment, acpt_rate, vfe, vfe_grad = craft_step(key, samples, log_weights, 
                                                                  flow_apply, params, 
                                                                  kernel, log_density, 
                                                                  beta, beta_prev, step, 
                                                                  threshold, 
                                                                  loss_val_and_grad)
    return (new_samples, new_log_weights, beta), (log_evidence_increment, acpt_rate, vfe, vfe_grad)
  
  initial_state = (initial_samples, initial_log_weights, 0.)
  keys = jax.random.split(key, num=num_temps-1)
  steps = jnp.arange(1, num_temps)
  if not betas:
    betas = steps/(num_temps-1)
  chex.assert_shape(betas, (num_temps-1,))
  per_step_input = (keys, transition_params, betas, steps)
  (final_samples, final_log_weights, _), \
  (log_evidence_increments, acpt_rate, vfes, vfe_grads) = jax.lax.scan(scan_step, 
                                                                        initial_state,
                                                                        per_step_input)
  log_evidence_estimate = jnp.sum(log_evidence_increments)
  total_vfe = jnp.sum(vfes)

  def update_step_params(step_params, step_grad, step_opt_state):
    step_updates, new_opt_state = opt.update(step_grad, step_opt_state)
    new_step_params = optax.apply_updates(step_params, step_updates)
    return new_step_params, new_opt_state
  
  final_transition_params, final_opt_states = jax.vmap(update_step_params)(transition_params,
                                                                            vfe_grads, 
                                                                            opt_states)

  return final_samples, final_log_weights, acpt_rate, \
    final_transition_params, final_opt_states, log_evidence_estimate, total_vfe

def time_embedded_craft_loop(key: jax.Array, sampler: InitialDensitySampler, 
                             flow_apply: Callable[[dict, jax.Array, float, float], 
                                                  Tuple[jax.Array, jax.Array]], 
                             params: dict, opt: optax.GradientTransformation, 
                             opt_state: optax.OptState, kernel: HMCKernel, 
                             log_density: LogDensityByTemp, threshold: float, 
                             num_temps: int, 
                             loss_val_and_grad: Callable[[jax.Array, jax.Array, 
                                                          dict, float, float], 
                                                         Tuple[float, jax.Array]], 
                             betas: jax.Array | None = None) -> Tuple[jax.Array, jax.Array, jax.Array, 
                                                                      dict, optax.OptState, float, float]:
  """A single run of the time-embedded CRAFT algorithm, with flow parameter updates.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  sampler : InitialDensitySampler
    A callable that takes a jax PRNG key and returns an array 
    of particles following the initial distribution.
  flow_apply : Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params, samples, and the 
    current and previous annealing temperatures to transport the 
    input samples by the underlying flow model. 
  params : dict
    The parameters of the shared flow model across temperatures.
  opt : optax.GradientTransformation
    The optimizer of the flow model.
  opt_state : optax.OptState
    The optimizer state of the shared flow model across temperatures.
  kernel : HMCKernel
    The HMC kernel to be applied in each mutation step.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  threshold : float
    The ESS threshold to trigger resampling.
  num_temps : int
    The number of annealing temperatures to be used.
  loss_val_and_grad : Callable[[jax.Array, jax.Array, dict, float, float], 
                               Tuple[float, jax.Array]]
    A function that takes in an array of samples, their log weights, 
    parameters of the flow, and the current and previous annealing 
    temperatures, and returns the value and gradient of the 
    variational free energy.
  betas : jax.Array | None = None
    An optional argument for the array of temperatures to be used by 
    the CRAFT algorithm. If not specified, defaults to a geometric 
    annealing schedule.

  Returns
  -------
  final_samples : jax.Array
    The array of particles produced after the CRAFT algorithm.
  final_log_weights : jax.Array
    An array containing the log weights of the particles in final_samples.
  acpt_rate : jax.Array
    An array containing the acceptance rates of the HMC kernel 
    at each annealing temperature.
  new_params : dict
    The updated flow parameters after learning from this CRAFT iteration.
  new opt_state : optax.OptState
    The updated optimization state of the flow parameters.
  log_evidence_estimate : float
    The log evidence estimate produced from this run of CRAFT (with 
    pre-update transition_params used for the flows.)
  total_vfe : float
    The total loss of the flow models from this run of CRAFT.
  """
  key, key_ = jax.random.split(key)
  initial_samples = sampler(key_)
  initial_log_weights = -jnp.log(sampler.num_particles) * jnp.ones(sampler.num_particles)

  def scan_step(state, per_step_input):
    samples, log_weights, beta_prev = state
    key, beta, step = per_step_input
    flow_apply_partial = jax.tree_util.Partial(flow_apply, beta=beta, 
                                               beta_prev=beta_prev)
    new_samples, new_log_weights, \
    log_evidence_increment, acpt_rate, vfe, vfe_grad = craft_step(key, samples, log_weights, 
                                                                  flow_apply_partial, params, 
                                                                  kernel, log_density, 
                                                                  beta, beta_prev, step, 
                                                                  threshold, 
                                                                  loss_val_and_grad)
    return (new_samples, new_log_weights, beta), (log_evidence_increment, acpt_rate, vfe, vfe_grad)
  
  initial_state = (initial_samples, initial_log_weights, 0.)
  keys = jax.random.split(key, num=num_temps-1)
  steps = jnp.arange(1, num_temps)
  if not betas:
    betas = steps/(num_temps-1)
  chex.assert_shape(betas, (num_temps-1,))
  per_step_input = (keys, betas, steps)
  (final_samples, final_log_weights, _), \
  (log_evidence_increments, acpt_rate, vfes, vfe_grads) = jax.lax.scan(scan_step, 
                                                                        initial_state,
                                                                        per_step_input)
  log_evidence_estimate = jnp.sum(log_evidence_increments)
  total_vfe = jnp.sum(vfes)
  total_grad = jax.tree_map(lambda x: jnp.sum(x, axis=0), vfe_grads)

  updates, new_opt_state = opt.update(total_grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  return final_samples, final_log_weights, acpt_rate, \
    new_params, new_opt_state, log_evidence_estimate, total_vfe

def simplify_craft_loop(sampler: InitialDensitySampler, 
                        flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
                        Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]], 
                        opt: optax.GradientTransformation, 
                        kernel: HMCKernel, 
                        log_density: LogDensityByTemp, 
                        threshold: float, 
                        num_temps: int, 
                        loss_val_and_grad: Callable[[jax.Array, jax.Array, 
                                                     dict, float, float], 
                                                     Tuple[float, jax.Array]], 
                        betas: jax.Array | None = None,
                        embed_time: bool = False
                        ) -> Callable[[jax.Array, dict, optax.OptState], 
                                      Tuple[jax.Array, jax.Array, jax.Array, 
                                            dict, optax.OptState, float, float]]:
  """Simplifies the craft_loop function signature by freezing arguments that 
  remain constant through the CRAFT training and evaluation.
  
  Parameters
  ----------
  sampler : InitialDensitySampler
    A callable that takes a jax PRNG key and returns an array 
    of particles following the initial distribution.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
               Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model. If embed_time is true, 
    then this function also takes the current and previous annealing 
    temperatures as input.
  opt : optax.GradientTransformation
    The optimizer of the flow model.
  kernel : HMCKernel
    The HMC kernel to be applied in each mutation step.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  threshold : float
    The ESS threshold to trigger resampling.
  num_temps : int
    The number of annealing temperatures to be used.
  loss_val_and_grad : Callable[[jax.Array, jax.Array, dict, float, float], 
                               Tuple[float, jax.Array]]
    A function that takes in an array of samples, their log weights, 
    parameters of the flow, and the current and previous annealing 
    temperatures, and returns the value and gradient of the 
    variational free energy.
  betas : jax.Array | None = None
    An optional argument for the array of temperatures to be used by 
    the CRAFT algorithm. If not specified, defaults to a geometric 
    annealing schedule.
  embed_time : bool = False
    A boolean that indicates whether to share parameters across 
    the temperatures and embed the annealing temperature into 
    the flow.

  Returns
  -------
  simplified_craft : Callable[[jax.Array, dict, optax.OptState], 
                              Tuple[jax.Array, jax.Array, jax.Array, 
                                    dict, optax.OptState, float, float]]
    A simplified craft_loop with most arguments frozen.
  """
  if not embed_time:
    def simplified_craft(key, transition_params, opt_states):
      final_samples, final_log_weights, acpt_rate, \
      final_transition_params, final_opt_states, \
      log_evidence_estimate, total_vfe = craft_loop(key, sampler, flow_apply, 
                                                    transition_params, opt, 
                                                    opt_states, kernel, 
                                                    log_density, threshold, 
                                                    num_temps, loss_val_and_grad, 
                                                    betas)
      return final_samples, final_log_weights, acpt_rate, \
            final_transition_params, final_opt_states, \
            log_evidence_estimate, total_vfe
  else:
    def simplified_craft(key, params, opt_state):
      final_samples, final_log_weights, acpt_rate, \
      final_params, final_opt_state, \
      log_evidence_estimate, total_vfe = time_embedded_craft_loop(key, sampler, 
                                                                  flow_apply, 
                                                                  params, opt, 
                                                                  opt_state, 
                                                                  kernel, 
                                                                  log_density, 
                                                                  threshold, 
                                                                  num_temps, 
                                                                  loss_val_and_grad, 
                                                                  betas)
      return final_samples, final_log_weights, acpt_rate, \
            final_params, final_opt_state, \
            log_evidence_estimate, total_vfe
    
  return simplified_craft

def apply(key: jax.Array, sampler: InitialDensitySampler, 
          flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
          Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]], 
          init_flow_params: dict,
          opt: optax.GradientTransformation, 
          log_density: LogDensityByTemp, 
          kernel: HMCKernel, 
          num_temps: int, 
          threshold: float, 
          num_train_iters: int, 
          betas: jax.Array | None, 
          report_interval: int,
          embed_time: bool
          ) -> Tuple[jax.Array, jax.Array, jax.Array, 
                     float, jax.Array, jax.Array]:
  """Applies the CRAFT algorithm.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  sampler : InitialDensitySampler
    A callable that takes a jax PRNG key and returns an array 
    of particles following the initial distribution.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
               Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model. If embed_time is true, 
    then this function also takes the current and previous annealing 
    temperatures as input.
  init_flow_params : dict
    An initial set of parameters for the flow model of a single temperature 
    step, to be used as the initial parameters for the flow models of all 
    CRAFT training and evaluation runs.
  opt : optax.GradientTransformation
    The optimizer for the flow models.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  kernel : HMCKernel
    The HMC kernel to be applied in each mutation step.
  num_temps : int
    The number of annealing temperatures to be used for training 
    and evaluation runs of CRAFT.
  threshold : float
    The ESS threshold to trigger resampling.
  num_train_iters : int
    The number of CRAFT training iterations to be performed during 
    the training phase.
  betas : jax.Array | None
    An optional argument for the array of temperatures to be used by 
    the CRAFT algorithm. If None, defaults to a geometric annealing 
    schedule.
  report_interval : int
    The number of temperatures before reporting training status again. 
  embed_time : bool
    A boolean that indicates whether to share parameters across 
    the temperatures and embed the annealing temperature into 
    the flow.

  Returns
  -------
  final_samples : jax.Array
    The array of particles produced after the CRAFT algorithm.
  final_log_weights : jax.Array
    An array containing the log weights of the particles in final_samples.
  acpt_rate : jax.Array
    An array containing the acceptance rates of the HMC kernel at each 
    annealing temperature of the evaluation run of CRAFT.
  log_evidence_estimate : float
    The log evidence estimate produced from the evaluation run of CRAFT.
  vfe_history : jax.Array
    An array recording the total loss of the flow model through each 
    run of CRAFT.
  log_evidence_history : jax.Array
    An array recording the log evidence estimate obtained through 
    each run of CRAFT.
  """
  # initialize starting variables
  key, key_ = jax.random.split(key)
  def loss_fn(samples, log_weights, flow_params, beta, beta_prev):
    return estimate_free_energy(samples, log_weights, flow_apply, 
                                flow_params, log_density, beta, 
                                beta_prev, embed_time)
  loss_val_and_grad = jax.value_and_grad(loss_fn, argnums=2)

  init_opt_state = opt.init(init_flow_params)

  if embed_time:
    params = init_flow_params
    opt_state = init_opt_state
  else:
    repeater = lambda x: jnp.repeat(x[None], num_temps-1, axis=0)
    opt_state = jax.tree_util.tree_map(repeater, init_opt_state)
    params = jax.tree_util.tree_map(repeater, init_flow_params)

  # jit step
  logging.info('Jitting step...')
  jitted_craft_loop = jax.jit( simplify_craft_loop(sampler, flow_apply, opt, 
                                                   kernel, log_density, 
                                                   threshold, num_temps, 
                                                   loss_val_and_grad, betas, 
                                                   embed_time) )
  logging.info('Performing initial step redundantly for accurate timing...')
  initial_start_time = time()
  jitted_craft_loop(key_, params, opt_state)
  initial_finish_time = time()
  initial_time_diff = initial_finish_time - initial_start_time
  logging.info(f'Initial step time / seconds: {initial_time_diff}')

  # training step
  vfe_history = []
  log_evidence_history = []
  logging.info('Launching training...')
  start_time = time()
  for step in range(num_train_iters+1):
    key, key_ = jax.random.split(key)
    final_samples, final_log_weights, acpt_rate, \
    params, opt_state, log_evidence, total_vfe = jitted_craft_loop(key_, 
                                                                   params, 
                                                                   opt_state)
    vfe_history.append(total_vfe)
    log_evidence_history.append(log_evidence)
    if step % report_interval == 0:
      logging.info(f"Step {step:04d}: vfe {total_vfe:.5f} \t log evidence {log_evidence:.5f} \t acceptance rate {jnp.mean(acpt_rate):.5f}")
  finish_time = time()
  train_time_diff = finish_time - start_time

  # end-of-training info dump
  logging.info(f"Training time / seconds : {train_time_diff}")
  logging.info(f"Log evidence estimate : {log_evidence}")

  vfe_history = jnp.array(vfe_history)
  log_evidence_history = jnp.array(log_evidence_history)

  return final_samples, final_log_weights, acpt_rate, \
         log_evidence, vfe_history, log_evidence_history