"""Implementation of the Annealed Flow Transport (AFT) sampler algorithm."""

import jax
import jax.numpy as jnp
import optax
from absl import logging
from time import time
from typing import Tuple, NamedTuple, Callable
from .utils.aft_types import InitialDensitySampler, LogDensityByTemp
from .utils.hmc import HMCKernel
from .utils.smc_utils import update_step_with_flow, estimate_free_energy
from .utils.smc_utils import estimate_free_energy_with_time_embedding

class SamplesTuple(NamedTuple):
  """Container for the train, validation and test
    batches of particles.

  Attributes
  ----------
  train_samples : jax.Array
    A rank 2 array of training particles.
  validation_samples : jax.Array
    A rank 2 array of validation particles.
  test_samples : jax.Array
    A rank 2 array of test particles.
  """
  train_samples: jax.Array
  validation_samples: jax.Array
  test_samples: jax.Array

class LogWeightsTuple(NamedTuple):
  """Container for the log weights of the train, 
  validation and test batches of particles.
  
  Attributes
  ----------
  train_log_weights : jax.Array
    A rank 1 array of log weights of the training particles.
  validation_log_weights : jax.Array
    A rank 1 array of log weights of the validation particles.
  test_log_weights : jax.Array
    A rank 1 array of log weights of the test particles.
  """
  train_log_weights: jax.Array
  validation_log_weights: jax.Array
  test_log_weights: jax.Array

def initialize_particle_tuple(key: jax.Array, 
                              sampler: InitialDensitySampler, 
                              train_sampler: InitialDensitySampler
                              ) -> Tuple[SamplesTuple, LogWeightsTuple]:
  """Initialize the training, validation, and test samples, with their 
  corresponding log weights.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  sampler : InitialDensitySampler
    A function to sample the test particles.
  train_sampler : InitialDensitySampler
    A function to sample the training and validation particles.

  Returns
  -------
  samples_tuple : SamplesTuple
    A named tuple containing the training, validation, and test 
    particles.
  log_weights_tuple : LogWeightsTuple
    A named tuple containing the log weights of the training, 
    validation, and test particles.
  """
  keys = jax.random.split(key, num=3)

  batch_sizes = (train_sampler.num_particles, 
                 train_sampler.num_particles, 
                 sampler.num_particles)
  samples_tuple = SamplesTuple(train_sampler(keys[0]),
                               train_sampler(keys[1]),
                               sampler(keys[2]))
  log_weights_tuple = LogWeightsTuple(*[-jnp.log(batch) * 
                                        jnp.ones(batch) for batch in batch_sizes])
  return samples_tuple, log_weights_tuple

def update_step(key: jax.Array, samples_tuple: SamplesTuple, 
                log_weights_tuple: LogWeightsTuple, 
                flow_apply: Callable[[dict, jax.Array], 
                                     Tuple[jax.Array, jax.Array]], 
                flow_params: dict, kernel: HMCKernel, 
                log_density_by_temp: LogDensityByTemp, beta: float, 
                beta_prev: float, step: int, threshold: float
                ) -> Tuple[SamplesTuple, LogWeightsTuple, float, float]:
  """Updates the collections of particles and their log weights 
  through one aft iteration, assuming the flows are already learned.
  
  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  samples_tuple : SamplesTuple
    A named tuple containing the training, validation, and test 
    particles.
  log_weights_tuple : LogWeightsTuple
    A named tuple containing the log weights of the training, 
    validation, and test particles.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  kernel : HMCKernel
    The HMC Kernel to be applied in the MCMC step.
  log_density_by_temp : LogDensityByTemp
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

  Returns
  -------
  samples_tuple : SamplesTuple
    A named tuple containing the updated training, validation, and 
    test particles.
  log_weights_tuple : LogWeightsTuple
    A named tuple containing the updated log weights of the training, 
    validation, and test particles.
  log_evidence_increment : float
    The estimate of log Z_t - log Z_{t-1}, where Z_t is the normalizing 
    constant of the t'th bridging distribution.
  acpt_rate : float
    Average acceptance rate of all HMC moves in this batch of particles.
  """
  keys = jax.random.split(key, num=3)

  new_samples_list = []
  new_log_weights_list = []

  for subkey, samples, log_weights in zip(keys, 
                                          samples_tuple, 
                                          log_weights_tuple):
    new_samples, new_log_weights, \
    log_evidence_increment, acpt_rate = update_step_with_flow(subkey, samples, 
                                                              log_weights, flow_apply, 
                                                              flow_params, log_density_by_temp, 
                                                              beta, beta_prev, kernel, 
                                                              threshold, step)
    new_samples_list.append(new_samples)
    new_log_weights_list.append(new_log_weights)
  
  samples_tuple = SamplesTuple(*new_samples_list)
  log_weights_tuple = LogWeightsTuple(*new_log_weights_list)

  return samples_tuple, log_weights_tuple, log_evidence_increment, acpt_rate

def flow_train_step(train_samples: jax.Array, validation_samples: jax.Array, 
                    train_log_weights: jax.Array, validation_log_weights: jax.Array, 
                    params: dict, opt_state: optax.OptState, beta: float, 
                    beta_prev: float, best_val_loss: float, best_params: dict, 
                    best_opt_state: optax.OptState, opt: optax.GradientTransformation, 
                    loss_val_and_grad: Callable[[jax.Array, jax.Array, 
                                                 dict, float, float], 
                                                 Tuple[float, jax.Array]]
                    ) -> Tuple[dict, optax.GradientTransformation, dict, 
                               float, optax.OptState, float]:
  """One iteration of the training step for one temperature of the AFT algorithm.

  Parameters
  ----------
  train_samples : jax.Array
    The array of training particles.
  validation_samples : jax.Array
    The array of validation particles.
  train_log_weights : jax.Array
    The array of log weights of the training particles.
  validation_log_weights : jax.Array
    The array of log weights of the validation particles.
  params : dict
    The parameters of the current flow.
  opt_state : optax.OptState
    The current optimization state.
  beta : float
    The current annealing temperature.
  beta_prev : float
    The previous annealing temperature.
  best_val_loss : float
    The best validation loss obtained in the current training loop.
  best_params : dict
    The parameters of the flow that produced the best validation loss 
    in the current training loop.
  best_opt_state : optax.OptState
    The optimization state that produced the best validation loss 
    in the current training loop.
  opt : optax.GradientTransformation
    The optimizer used in the AFT algorithm.
  loss_val_and_grad : Callable[[jax.Array, jax.Array, dict, float, float], 
                               Tuple[float, jax.Array]]
    A function that takes in an array of samples, their log weights, 
    parameters of the flow, and the current and previous annealing 
    temperatures, and returns the value and gradient of the 
    variational free energy.

  Returns
  -------
  new_params : dict
    The parameters of the flow after one iteration of training.
  new_opt_state : optax.OptState
    The optimization state of the flow after one iteration of training.
  new_best_params : dict
    The parameters of the flow that produced the best validation loss 
    in the current training loop so far.
  new_best_val_loss : float
    The best validation loss obtained in the current training loop so far.
  new_best_opt_state : optax.OptState
    The optimization state that produced the best validation loss in 
    the current training loop so far.
  train_loss : float
    The training loss at the start of this training iteration.
  """
  # compute loss values and gradients
  train_loss, train_grad = loss_val_and_grad(train_samples, train_log_weights, 
                                             params, beta, beta_prev)
  validation_loss, _ = loss_val_and_grad(validation_samples, 
                                         validation_log_weights, 
                                         params, beta, beta_prev)
  
  # update best validation loss and param set
  new_best_val_loss, \
  new_best_params, new_best_opt_state = jax.lax.cond(validation_loss < best_val_loss, 
                                                     lambda _: (validation_loss, params, opt_state), 
                                                     lambda _: (best_val_loss, best_params, best_opt_state), 
                                                     operand=None)
  
  # apply gradients for next training step
  updates, new_opt_state = opt.update(train_grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  return new_params, new_opt_state, new_best_params, \
         new_best_val_loss, new_best_opt_state, train_loss

def get_train_step(train_samples: jax.Array, validation_samples: jax.Array, 
                   train_log_weights: jax.Array, validation_log_weights: jax.Array, 
                   beta: float, beta_prev: float, opt: optax.GradientTransformation, 
                   loss_val_and_grad: Callable[[jax.Array, jax.Array, 
                                                 dict, float, float], 
                                                 Tuple[float, jax.Array]]
                   ) -> Callable[[Tuple[dict, optax.OptState, dict, float, optax.OptState], None], 
                                 Tuple[Tuple[dict, optax.OptState, dict, float, optax.OptState], 
                                       jax.Array]]:
  """Simplifies the flow_train_step function signature by freezing arguments that 
  remain constant through the training loop within the annealing temperature.

  Parameters
  ----------
  train_samples : jax.Array
    The array of training particles.
  validation_samples : jax.Array
    The array of validation particles.
  train_log_weights : jax.Array
    The array of log weights of the training particles.
  validation_log_weights : jax.Array
    The array of log weights of the validation particles.
  beta : float
    The current annealing temperature.
  beta_prev : float
    The previous annealing temperature.
  opt : optax.GradientTransformation
    The optimizer used in the AFT algorithm.
  loss_val_and_grad : Callable[[jax.Array, jax.Array, dict, float, float], 
                               Tuple[float, jax.Array]]
    A function that takes in an array of samples, their log weights, 
    parameters of the flow, and the current and previous annealing 
    temperatures, and returns the value and gradient of the 
    variational free energy.

  Returns
  -------
  train_step : Callable[[Tuple[dict, optax.OptState, dict, float, optax.OptState], None], 
                        Tuple[Tuple[dict, optax.OptState, dict, float, optax.OptState], 
                              jax.Array]]
    Scannable flow_train_step with simplified signature.
  """
  def train_step(state, unused_input):
    params, opt_state, best_params, best_val_loss, best_opt_state = state
    new_params, new_opt_state, new_best_params, \
    new_best_val_loss, new_best_opt_state, train_loss = flow_train_step(train_samples, 
                                                                        validation_samples,
                                                                        train_log_weights, 
                                                                        validation_log_weights, 
                                                                        params, opt_state, beta, 
                                                                        beta_prev, best_val_loss, 
                                                                        best_params, best_opt_state, 
                                                                        opt, loss_val_and_grad)
    return (new_params, new_opt_state, new_best_params, 
            new_best_val_loss, new_best_opt_state), train_loss
  return train_step

def aft_step(key: jax.Array, samples_tuple: SamplesTuple, 
             log_weights_tuple: LogWeightsTuple, beta: float, beta_prev: float, 
             flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
             Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]], 
             flow_params: dict, kernel: HMCKernel, log_density: LogDensityByTemp, 
             step: int, threshold: float, num_train_iters: int, 
             opt_state:optax.OptState, opt: optax.GradientTransformation, 
             loss_val_and_grad: Callable[[jax.Array, jax.Array, 
                                                 dict, float, float], 
                                                 Tuple[float, jax.Array]], 
             embed_time: bool = False
             ) -> Tuple[jax.Array, jax.Array, float, float, 
                        float, dict, optax.OptState, jax.Array]:
  """A single iteration step of the AFT algorithm.
  
  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  samples_tuple : SamplesTuple
    A NamedTuple containing the training, validation, and test particles 
    for this annealing temperature.
  log_weights_tuple : LogWeightsTuple
    A NamedTuple containing the log weights of the training, validation, 
    and test particles for this annealing temperature.
  beta : float
    The current annealing temperature.
  beta_prev : float
    The previous annealing temperature.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
               Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model. If embed_time is true, 
    then this function also takes the current and previous annealing 
    temperatures as input.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  kernel : HMCKernel
    The HMC Kernel to be applied in the MCMC step.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  step : int
    The current iteration time step.
  threshold : float
    The ESS threshold to trigger resampling.
  num_train_iters : int
    The number of iterations the training loop should run for.
  opt_state: optax.OptState
    The current optimization state.
  opt : optax.GradientTransformation
    The optimizer of the flow model for this AFT algorithm.
  loss_val_and_grad : Callable[[jax.Array, jax.Array, dict, float, float], 
                               Tuple[float, jax.Array]]
    A function that takes in an array of samples, their log weights, 
    parameters of the flow, and the current and previous annealing 
    temperatures, and returns the value and gradient of the 
    variational free energy.
  embed_time : bool = False
    A boolean that indicates whether to share parameters across 
    the temperatures and embed the annealing temperature into 
    the flow.

  Returns
  -------
  samples_tuple : SamplesTuple
    A named tuple containing the updated training, validation, and 
    test particles.
  log_weights_tuple : LogWeightsTuple
    A named tuple containing the updated log weights of the training, 
    validation, and test particles.
  log_evidence_increment : float
    The estimate of log Z_t - log Z_{t-1}, where Z_t is the normalizing 
    constant of the t'th bridging distribution.
  acpt_rate : float
    Average acceptance rate of all HMC moves in this batch of particles.
  best_val_loss : float
    The validation loss of the final flow used in this annealing 
    temperature.
  best_params : dict
    The parameters of the flow that produced the best validation loss 
    in the current training loop so far.
  best_opt_state : optax.OptState
    The optimization state that produced the best validation loss in 
    the current training loop so far.
  train_loss : jax.Array
    The training loss history of the training phase in this annealing 
    temperature.
  """
  train_step = get_train_step(samples_tuple[0], samples_tuple[1], 
                              log_weights_tuple[0], log_weights_tuple[1], 
                              beta, beta_prev, opt, loss_val_and_grad)

  # flow training
  (_, _, best_params, best_val_loss, best_opt_state), train_loss = jax.lax.scan(train_step, 
                                                                                (flow_params, 
                                                                                 opt_state, 
                                                                                 flow_params, jnp.inf, 
                                                                                 opt_state), 
                                                                                None, num_train_iters)
      
  # samples and log weights updates
  if embed_time:
    flow_apply = jax.tree_util.Partial(flow_apply, beta=beta, beta_prev=beta_prev)
  samples_tuple, log_weights_tuple, log_evidence_increment, acpt_rate = update_step(key, 
                                                                                    samples_tuple, 
                                                                                    log_weights_tuple, 
                                                                                    flow_apply, 
                                                                                    best_params, 
                                                                                    kernel, 
                                                                                    log_density, 
                                                                                    beta, beta_prev, 
                                                                                    step, threshold)
  
  return samples_tuple, log_weights_tuple, log_evidence_increment, acpt_rate, \
         best_val_loss, best_params, best_opt_state, train_loss

def apply(key: jax.Array, log_density: LogDensityByTemp, sampler: InitialDensitySampler, 
          train_sampler: InitialDensitySampler, kernel: HMCKernel, 
          flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
          Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]],
          params: dict, opt: optax.GradientTransformation, threshold: float,
          num_train_iters: int, num_temps: int, betas: jax.Array | None, 
          report_interval: int, embed_time: bool, refresh_opt_state: bool
          ) -> Tuple[jax.Array, jax.Array, float, jax.Array, jax.Array, jax.Array]:
  """Applies the AFT algorithm.
  
  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  sampler : InitialDensitySampler
    A callable that takes a jax PRNG key as input, and outputs an array of 
    shape (num_particles, particle_dim) containing randomly generated 
    particles under the initial distribution.
  train_sampler : InitialDensitySampler
    A callable that takes a jax PRNG key as input, and outputs an array of 
    shape (num_train_particles, particle_dim) containing randomly generated 
    particles used for training the normalizing flow. The distribution of 
    this sampler should be the same as that of sampler.
  kernel : HMCKernel
    The HMC kernel to be used throughout the SMC algorithm.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
               Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model. If embed_time is true, 
    then this function also takes the current and previous annealing 
    temperatures as input.
  params : dict
    The initial parameters of the flow model of flow_apply.
  opt : optax.GradientTransformation
    The optimizer used to train the flow models.
  threshold : float
    The ESS threshold to trigger resampling.
  num_train_iters : int
    The number of iterations that the training loop should run for each 
    annealing temperature.
  num_temps : int
    The total number of annealing temperatures for the AFT.
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
  refresh_opt_state : bool
    A boolean that indicates whether to refresh the optimization 
    state after a training loop.

  Returns
  -------
  final_samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    final positions of the particles.
  final_log_weights : jax.Array
    An array of shape (num_particles,) containing the log weights of 
    the particles in final_sample.
  log_evidence_estimate : float
    An estimate of the log evidence of the target density.
  acpt_rate_history : jax.Array
    An array of shape (num_temps-1,) containing the average acceptance 
    rate of the HMC kernel for each temperature.
  val_loss_history : jax.Array
    An array containing the validation losses for the final parameters 
    used the flow model for each temperature.
  train_loss_history : jax.Array
    An array containing the training losses for the training loops in 
    each temperature.
  """
  # initialize starting variables
  key, key_ = jax.random.split(key)
  if not betas:
    betas = jnp.arange(1, num_temps)/(num_temps-1)
  samples_tuple, log_weights_tuple = initialize_particle_tuple(key_, sampler, train_sampler)

  def loss_fn(samples, log_weights, flow_params, beta, beta_prev):
    return estimate_free_energy(samples, log_weights, flow_apply, 
                                flow_params, log_density, beta, 
                                beta_prev, embed_time)
  
  loss_val_and_grad = jax.value_and_grad(loss_fn, argnums=2)

  def specified_aft_step(key, samples_tuple, log_weights_tuple, beta, 
                         beta_prev, params, opt_state, step):
    return aft_step(key, samples_tuple, log_weights_tuple, beta, beta_prev, 
                    flow_apply, params, kernel, log_density, step, threshold, 
                    num_train_iters, opt_state, opt, loss_val_and_grad, embed_time)
  
  opt_state = opt.init(params)

  # jit step
  logging.info('Jitting step...')
  jitted_aft_step = jax.jit( specified_aft_step )
  logging.info('Performing initial step redundantly for accurate timing...')
  initial_start_time = time()
  jitted_aft_step(key_, samples_tuple, log_weights_tuple, 
                  0.1, 0, params, opt_state, 1)
  initial_finish_time = time()
  initial_time_diff = initial_finish_time - initial_start_time
  logging.info(f'Initial step time / seconds: {initial_time_diff}')

  # training step
  beta = 0.
  log_evidence = 0.
  acpt_rate_history = []
  val_loss_history = []
  train_loss_history = []
  logging.info('Launching training...')
  start_time = time()
  for step in range(1, num_temps):
    key, key_ = jax.random.split(key)
    beta_prev = beta
    beta = betas[step-1]
    samples_tuple, log_weights_tuple, log_evidence_increment, acpt_rate, \
    best_val_loss, best_params, best_opt_state, train_loss = jitted_aft_step(key_, samples_tuple, 
                                                                             log_weights_tuple, beta, 
                                                                             beta_prev, params, 
                                                                             opt_state, step)
    
    if embed_time:
      params = best_params

    if not refresh_opt_state:
      opt_state = best_opt_state

    log_evidence += log_evidence_increment
    acpt_rate_history.append(acpt_rate)
    val_loss_history.append(best_val_loss)
    train_loss_history.append(train_loss)
    if step % report_interval == 0:
      logging.info(f"Step {step:04d}: beta {beta:.5f} \t validation loss {best_val_loss:.5f} \t acceptance rate {acpt_rate:.5f}")
  finish_time = time()
  train_time_diff = finish_time - start_time

  # end-of-training info dump
  logging.info(f"Training time / seconds : {train_time_diff}")
  logging.info(f"Log evidence estimate : {log_evidence}")

  acpt_rate_history = jnp.array(acpt_rate_history)
  samples = samples_tuple[-1]
  log_weights = log_weights_tuple[-1]
  val_loss_history = jnp.array(val_loss_history)
  train_loss_history = jnp.array(train_loss_history)

  return samples, log_weights, log_evidence, acpt_rate_history, val_loss_history, train_loss_history