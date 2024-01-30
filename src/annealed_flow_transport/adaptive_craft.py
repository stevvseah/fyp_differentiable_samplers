"""Implementation of an adaptive variant of the CRAFT algorithm."""

import jax
import jax.numpy as jnp
from absl import logging
from time import time
import chex
import optax
from typing import Tuple, Callable
from .utils.aft_types import InitialDensitySampler, LogDensityByTemp
from .utils.hmc import HMCKernel, tree_add, tree_scalar_mul
from .utils.smc_utils import update_step_with_flow, estimate_free_energy
from .utils.smc_utils import adaptive_temp_search_with_flow
from .craft import craft_step

def ada_craft_step(key: jax.Array, samples: jax.Array, 
                   log_weights: jax.Array, params: dict, 
                   flow_apply: Callable[[dict, jax.Array, float, float], 
                                        Tuple[jax.Array, jax.Array]], 
                   kernel: HMCKernel, log_density: LogDensityByTemp, 
                   beta: float, step: int, threshold: float, 
                   loss_val_and_grad: Callable[[jax.Array, jax.Array, 
                                                dict, float, float], 
                                               Tuple[float, jax.Array]], 
                   num_search_iters: int, adaptive_threshold: float
                   ) -> Tuple[jax.Array, jax.Array, float, 
                              float, float, jax.Array, float]:
  """A temperature step of adaptive CRAFT that updates the input 
  samples and their log weights.
  
  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  samples : jax.Array
    An array containing the batch of particles.
  log_weights : jax.Array
    An array containing the log weights of the particles in samples.
  params : dict
    The parameters of the flow model of flow_apply.
  flow_apply : Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]]
    A function that takes as input params, samples, current 
    annealing temperature, and previous annealing temperature 
    to transport the input samples by the underlying flow model.
  kernel : HMCKernel
    The HMC Kernel to be applied in the MCMC step.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  beta : float
    The temperature at the current iteration.
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
  num_search_iters : int
    The number of iterations to run the bisection method search for 
    the next annealing temperature in the adaptive SMC algorithm.
  adaptive_threshold : float
    The target conditional effective sample size for the selection 
    of the next annealing temperature.

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
  beta : float
    The current annealing temperature as chosen by the adaptive 
    annealing temperature search.
  """
  step += 1
  beta_prev = beta
  beta = adaptive_temp_search_with_flow(samples, flow_apply, params, 
                                        log_weights, beta, log_density, 
                                        num_search_iters, 
                                        adaptive_threshold, True)

  flow_apply_partial = jax.tree_util.Partial(flow_apply, beta=beta, 
                                             beta_prev=beta_prev)

  new_samples, new_log_weights, log_evidence_increment, \
  acpt_rate, vfe, vfe_grad = craft_step(key, samples, log_weights, 
                                        flow_apply_partial, params, kernel, 
                                        log_density, beta, beta_prev, 
                                        step, threshold, loss_val_and_grad)

  return new_samples, new_log_weights, log_evidence_increment, \
         acpt_rate, vfe, vfe_grad, beta

def ada_craft_loop(key: jax.Array, sampler: InitialDensitySampler, 
                   flow_apply: Callable[[dict, jax.Array, float, float], 
                                        Tuple[jax.Array, jax.Array]], 
                   params: dict, opt: optax.GradientTransformation, 
                   opt_state: optax.OptState, kernel: HMCKernel, 
                   log_density: LogDensityByTemp, threshold: float, 
                   loss_val_and_grad: Callable[[jax.Array, jax.Array, 
                                                dict, float, float], 
                                                Tuple[float, jax.Array]], 
                   num_search_iters: int, adaptive_threshold: float, 
                   max_adaptive_num_temps: int
                   ) -> Tuple[jax.Array, jax.Array, jax.Array, dict, 
                              optax.OptState, float, float, jax.Array, int]:
  """A single run of the adaptive CRAFT algorithm with time embedded flow, 
  with flow parameter updates. 
  
  Note that if max_adaptive_num_temps is reached 
  before the final annealing temperature, the algorithm will terminate without 
  processing the samples at the last annealing temperature.

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
  loss_val_and_grad : Callable[[jax.Array, jax.Array, dict, float, float], 
                               Tuple[float, jax.Array]]
    A function that takes in an array of samples, their log weights, 
    parameters of the flow, and the current and previous annealing 
    temperatures, and returns the value and gradient of the 
    variational free energy.
  num_search_iters : int
    The number of iterations to run the bisection method search for 
    the next annealing temperature in the adaptive SMC algorithm.
  adaptive_threshold : float
    The target conditional effective sample size for the selection 
    of the next annealing temperature.
  max_adaptive_num_temps : int
    The maximum number of temperatures to allow the adaptive algorithm 
    to proceed for, after which the algorithm terminates with whatever 
    results were already obtained.

  Returns
  -------
  new_samples : jax.Array
    The array of particles produced after the CRAFT algorithm.
  new_log_weights : jax.Array
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
  betas : jax.Array
    The array of annealing temperatures chosen by the adaptive temperature 
    search. Its shape is fixed to (max_adaptive_num_temps,), where values 
    beyond 1 are set to jnp.nan for ease of debugging.
  final_step : int
    The final time step index where the algorithm was actually training. 
    This can be useful for tuning max_adaptive_num_temps and adaptive_threshold.
  """
  key, key_ = jax.random.split(key)
  initial_samples = sampler(key_)
  initial_log_weights = -jnp.log(sampler.num_particles) * jnp.ones(sampler.num_particles)
  _, sample_grad = loss_val_and_grad(initial_samples, initial_log_weights, 
                                     params, 0.1, 0.)

  def train_step(state, key):
    samples, log_weights, beta, step = state
    step += 1
    new_samples, new_log_weights, log_evidence_increment, \
    acpt_rate, vfe, vfe_grad, beta = ada_craft_step(key, samples, log_weights, 
                                                    params, flow_apply, kernel, 
                                                    log_density, beta, step, 
                                                    threshold, loss_val_and_grad, 
                                                    num_search_iters, 
                                                    adaptive_threshold)
    return (new_samples, new_log_weights, beta, step), \
           (log_evidence_increment, acpt_rate, vfe, vfe_grad, beta)
  
  def stall_step(state, key):
    zero_grad = tree_scalar_mul(sample_grad, 0.)
    per_step_output = (0., jnp.nan, 0., zero_grad, jnp.nan)
    return state, per_step_output
  
  def scan_step(state, key):
    samples, log_weights, beta, step = state
    new_state, per_step_output = jax.lax.cond(beta < 1., train_step, 
                                              stall_step, state, key)
    return new_state, per_step_output
  
  initial_state = (initial_samples, initial_log_weights, 0., 0)
  keys = jax.random.split(key, num=max_adaptive_num_temps)
  (new_samples, new_log_weights, beta, final_step), \
  (log_evidence_increments, acpt_rate, vfes, 
   vfe_grads, betas) = jax.lax.scan(scan_step, initial_state, keys)
  log_evidence_estimate = jnp.sum(log_evidence_increments)
  total_vfe = jnp.sum(vfes)

  total_grad = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), vfe_grads)
  updates, new_opt_state = opt.update(total_grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  # def update_params_step(state, grad):
  #   params, opt_state = state
  #   updates, new_opt_state = opt.update(grad, opt_state)
  #   new_params = optax.apply_updates(params, updates)
  #   return (new_params, new_opt_state), None

  # (new_params, new_opt_state), _ = jax.lax.scan(update_params_step, 
  #                                               (params, opt_state), 
  #                                               vfe_grads)

  return new_samples, new_log_weights, acpt_rate, new_params, \
         new_opt_state, log_evidence_estimate, total_vfe, betas, final_step

def apply(key: jax.Array, sampler: InitialDensitySampler, 
          flow_apply: Callable[[dict, jax.Array, float, float], 
                               Tuple[jax.Array, jax.Array]], 
          params: dict, opt: optax.GradientTransformation, 
          kernel: HMCKernel, log_density: LogDensityByTemp, 
          threshold: float, num_search_iters: int, 
          adaptive_threshold: float, max_adaptive_num_temps: int, 
          num_train_iters: int, report_interval: int
          ) -> Tuple[jax.Array, jax.Array, jax.Array, float, 
                     jax.Array, jax.Array, jax.Array, int]:
  """Applies the adaptive CRAFT algorithm.

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
  kernel : HMCKernel
    The HMC kernel to be applied in each mutation step.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  threshold : float
    The ESS threshold to trigger resampling.
  num_search_iters : int
    The number of iterations to run the bisection method search for 
    the next annealing temperature in the adaptive SMC algorithm.
  adaptive_threshold : float
    The target conditional effective sample size for the selection 
    of the next annealing temperature.
  max_adaptive_num_temps : int
    The maximum number of temperatures to allow the adaptive algorithm 
    to proceed for, after which the algorithm terminates with whatever 
    results were already obtained.
  num_train_iters : int
    The number of CRAFT training iterations to be performed during 
    the training phase.
  report_interval : int
    The number of temperatures before reporting training status again. 

  Returns
  -------
  final_samples : jax.Array
    The array of particles produced after the CRAFT algorithm.
  final_log_weights : jax.Array
    An array containing the log weights of the particles in final_samples.
  acpt_rate : jax.Array
    An array containing the acceptance rates of the HMC kernel at each 
    annealing temperature of the evaluation run of CRAFT.
  log_evidence : float
    The log evidence estimate produced from the evaluation run of CRAFT.
  vfe_history : jax.Array
    An array recording the total loss of the flow model through each 
    run of CRAFT.
  log_evidence_history : jax.Array
    An array recording the log evidence estimate obtained through 
    each run of CRAFT.
  betas : jax.Array
    The array of annealing temperatures chosen by the adaptive temperature 
    search. Its shape is fixed to (max_adaptive_num_temps,), where values 
    beyond 1 are set to jnp.nan for ease of debugging.
  final_step : int
    The final time step index where the algorithm was actually training. 
    This can be useful for tuning max_adaptive_num_temps and adaptive_threshold.
  """
  # initialize starting variables
  key, key_ = jax.random.split(key)
  def loss_fn(samples, log_weights, flow_params, beta, beta_prev):
    return estimate_free_energy(samples, log_weights, flow_apply, 
                                flow_params, log_density, beta, 
                                beta_prev, True)
  loss_val_and_grad = jax.value_and_grad(loss_fn, argnums=2)

  opt_state = opt.init(params)

  def simplified_loop(key, params, opt_state):
    return ada_craft_loop(key, sampler, flow_apply, params, opt, opt_state, kernel, 
                          log_density, threshold, loss_val_and_grad, num_search_iters, 
                          adaptive_threshold, max_adaptive_num_temps)

  # jit step
  logging.info('Jitting step...')
  initial_start_time = time()
  jitted_loop = jax.jit(simplified_loop)
  jitted_loop(key_, params, opt_state)
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
    final_samples, final_log_weights, acpt_rate, params, opt_state, \
    log_evidence, total_vfe, betas, final_step = jitted_loop(key_, params, opt_state)
    vfe_history.append(total_vfe)
    log_evidence_history.append(log_evidence)
    if step % report_interval == 0:
      logging.info(f"Step {step:04d}: vfe {total_vfe:.5f} \t \
log evidence {log_evidence:.5f} \t \
acceptance rate {jnp.nanmean(acpt_rate):.5f} \t \
final beta {betas[final_step-1]:.5f} \t \
num steps {final_step}")
  finish_time = time()
  train_time_diff = finish_time - start_time

  # end-of-training info dump
  logging.info(f"Training time / seconds : {train_time_diff}")
  logging.info(f"Log evidence estimate : {log_evidence}")

  vfe_history = jnp.array(vfe_history)
  log_evidence_history = jnp.array(log_evidence_history)

  return final_samples, final_log_weights, acpt_rate, log_evidence, \
         vfe_history, log_evidence_history, betas, final_step