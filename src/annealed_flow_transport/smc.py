"""Sequential Monte Carlo (SMC) sampler algorithm."""

import jax
import jax.numpy as jnp
from absl import logging
from time import time
from .utils.smc_utils import update_step_no_flow, adaptive_temp_search
from .utils.hmc import HMCKernel
from .utils.aft_types import LogDensityByTemp, InitialDensitySampler
from typing import Callable, Tuple

def get_smc_step(kernel: HMCKernel, 
                 log_density: LogDensityByTemp, 
                 threshold: float) -> Callable[[jax.Array, jax.Array, 
                                                jax.Array, float, 
                                                float, int], 
                                                Tuple[jax.Array, jax.Array, 
                                                      float, float]]:
  """Simplify the signature of update_step_no_flow by freezing the kernel, 
  log_density, and threshold arguments.

  Parameters
  ----------
  kernel : HMCKernel
    The HMC kernel to be used throughout the SMC algorithm.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  threshold : float
    The ESS threshold to trigger resampling.
  
  Returns
  -------
  smc_step : Callable[[jax.Array, jax.Array, jax.Array, float, float, int], 
                       Tuple[jax.Array, jax.Array, float, float]]
    Simplified update_step_no_flow function with kernel, log_density, and 
    threshold arguments frozen.
  """
  def smc_step(key: jax.Array, samples: jax.Array, 
               log_weights: jax.Array, beta: float, 
               beta_prev: float, step: int
               ) -> Tuple[jax.Array, jax.Array, 
                          float, float]:
    return update_step_no_flow(key, samples, log_weights, 
                               log_density, beta, beta_prev, 
                               kernel, threshold, step)
  return smc_step

def fast_smc_apply(key: jax.Array, log_density: LogDensityByTemp, 
                   initial_sampler: InitialDensitySampler, 
                   kernel: HMCKernel, threshold: float, 
                   num_temps: int
                   ) -> Tuple[jax.Array, jax.Array, float, jax.Array]:
  """Applies the SMC algorithm with a scanned smc_step.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  initial_sampler : InitialDensitySampler
    A callable that takes a jax PRNG key as input, and outputs an array of 
    shape (num_particles, particle_dim) containing randomly generated 
    particles under the initial distribution.
  kernel : HMCKernel
    The HMC kernel to be used throughout the SMC algorithm.
  threshold : float
    The ESS threshold to trigger resampling.
  num_temps : int
    The total number of annealing temperatures for the SMC.

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
  acpt_rate : jax.Array
    An array of shape (num_temps-1,) containing the average acceptance 
    rate of the HMC kernel for each temperature.
  """
  key, key_ = jax.random.split(key)
  batch_size = initial_sampler.num_particles

  samples = initial_sampler(key_)
  log_weights = -jnp.log(batch_size) * jnp.ones(batch_size)
  smc_step = get_smc_step(kernel, log_density, threshold)

  def scan_step(state, per_step_input):
    samples, log_weights, beta_prev = state
    beta, curr_key, curr_step = per_step_input
    new_samples, new_log_weights, log_evidence_increment, acpt_rate = smc_step(curr_key, 
                                                                               samples, 
                                                                               log_weights, 
                                                                               beta, 
                                                                               beta_prev, 
                                                                               curr_step)
    return (new_samples, new_log_weights, beta), (log_evidence_increment, acpt_rate)
  
  initial_state = (samples, log_weights, 0)
  steps = jnp.arange(1, num_temps)
  keys = jax.random.split(key, num=num_temps-1)
  betas = steps/(num_temps-1)
  per_step_input = (betas, keys, steps)
  (final_samples, final_log_weights, final_temp), (log_evidence_increments, acpt_rates) = jax.lax.scan(scan_step,
                                                                                                       initial_state,
                                                                                                       per_step_input)
  log_evidence_estimate = jnp.sum(log_evidence_increments)

  return final_samples, final_log_weights, log_evidence_estimate, acpt_rates

def apply(key: jax.Array, log_density: LogDensityByTemp, 
              initial_sampler: InitialDensitySampler, 
              kernel: HMCKernel, threshold: float, 
              num_temps: int, betas: jax.Array | None, 
              report_interval: int
              ) -> Tuple[jax.Array, jax.Array, float, jax.Array, float]:
  """Applies the SMC algorithm.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  initial_sampler : InitialDensitySampler
    A callable that takes a jax PRNG key as input, and outputs an array of 
    shape (num_particles, particle_dim) containing randomly generated 
    particles under the initial distribution.
  kernel : HMCKernel
    The HMC kernel to be used throughout the SMC algorithm.
  threshold : float
    The ESS threshold to trigger resampling.
  num_temps : int
    The total number of annealing temperatures for the SMC.
  betas : jax.Array | None
    An optional argument for the array of temperatures to be used by 
    the CRAFT algorithm. If specified as None, defaults to a geometric 
    annealing schedule.
  report_interval : int
    The number of temperatures before reporting training status again. 

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
  train_time_diff : float
    The time taken to perform sampling (without time spent on jit 
    compilation).
  """

  # initialize starting variables
  key, key_ = jax.random.split(key)
  if not betas:
    betas = jnp.arange(1, num_temps)/(num_temps-1)
  batch_size = initial_sampler.num_particles
  samples = initial_sampler(key_)
  log_weights = -jnp.log(batch_size) * jnp.ones(batch_size)

  # jit step
  logging.info('Jitting step...')
  smc_step = jax.jit( get_smc_step(kernel, log_density, threshold) )
  logging.info('Performing initial step redundantly for accurate timing...')
  initial_start_time = time()
  smc_step(key_, samples, log_weights, 0.1, 0, 1)
  initial_finish_time = time()
  initial_time_diff = initial_finish_time - initial_start_time
  logging.info(f'Initial step time / seconds: {initial_time_diff}')

  # training step
  beta = 0.
  log_evidence = 0.
  acpt_rate_history = []
  logging.info('Launching training...')
  start_time = time()
  for step in range(1, num_temps):
    key, key_ = jax.random.split(key)
    beta_prev = beta
    beta = betas[step-1]
    samples, log_weights, log_evidence_increment, acpt_rate = smc_step(key_, samples, 
                                                                       log_weights, beta, 
                                                                       beta_prev, step)
    log_evidence += log_evidence_increment
    acpt_rate_history.append(acpt_rate)
    if step % report_interval == 0:
      logging.info(f"Step {step:03d}: beta {beta:.5f} \t acceptance rate {acpt_rate:.5f}")
  finish_time = time()
  train_time_diff = finish_time - start_time

  # end of training info dump
  logging.info(f"Training time / seconds : {train_time_diff}")
  logging.info(f"Log evidence estimate : {log_evidence}")

  acpt_rate_history = jnp.array(acpt_rate_history)

  return samples, log_weights, log_evidence, acpt_rate_history, train_time_diff

def apply_adaptive(key: jax.Array, log_density: LogDensityByTemp, 
                   initial_sampler: InitialDensitySampler, 
                   kernel: HMCKernel, threshold: float, 
                   report_interval: int, num_search_iters: int, 
                   adaptive_threshold: float
                   ) -> Tuple[jax.Array, jax.Array, float, 
                              jax.Array, jax.Array, float]:
  """Applies an adaptive variant of SMC algorithm.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  initial_sampler : InitialDensitySampler
    A callable that takes a jax PRNG key as input, and outputs an array of 
    shape (num_particles, particle_dim) containing randomly generated 
    particles under the initial distribution.
  kernel : HMCKernel
    The HMC kernel to be used throughout the SMC algorithm.
  threshold : float
    The ESS threshold to trigger resampling.
  report_interval : int
    The number of temperatures before reporting training status again. 
  num_search_iters : int
    The number of iterations to run the bisection method search for 
    the next annealing temperature in the adaptive SMC algorithm.
  adaptive_threshold : float
    The target conditional effective sample size for the selection 
    of the next annealing temperature.

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
  beta_history : jax.Array
    An array containing the annealing temperatures adaptively selected.
  train_time_diff : float
    The time taken to perform sampling (without time spent on jit 
    compilation).
  """

  # initialize starting variables
  key, key_ = jax.random.split(key)
  batch_size = initial_sampler.num_particles
  samples = initial_sampler(key_)
  log_weights = -jnp.log(batch_size) * jnp.ones(batch_size)

  get_next_beta = jax.jit( jax.tree_util.Partial(adaptive_temp_search, 
                                                 log_density=log_density, 
                                                 num_search_iters=num_search_iters, 
                                                 threshold=adaptive_threshold) )

  # jit step
  logging.info('Jitting step...')
  smc_step = jax.jit( get_smc_step(kernel, log_density, threshold) )
  logging.info('Performing initial step redundantly for accurate timing...')
  initial_start_time = time()
  smc_step(key_, samples, log_weights, 0.1, 0, 1)
  get_next_beta(samples, log_weights, 0.)
  initial_finish_time = time()
  initial_time_diff = initial_finish_time - initial_start_time
  logging.info(f'Initial step time / seconds: {initial_time_diff}')

  # training step
  beta = 0.
  log_evidence = 0.
  step = 0
  acpt_rate_history = []
  beta_history = []
  logging.info('Launching training...')
  start_time = time()
  while beta < 1.:
    key, key_ = jax.random.split(key)
    step += 1
    beta_prev = beta
    beta = get_next_beta(samples, log_weights, beta)
    samples, log_weights, log_evidence_increment, acpt_rate = smc_step(key_, samples, 
                                                                       log_weights, beta, 
                                                                       beta_prev, step)
    log_evidence += log_evidence_increment
    acpt_rate_history.append(acpt_rate)
    beta_history.append(beta)
    if step % report_interval == 0:
      logging.info(f"Step {step:03d}: beta {beta:.5f} \t acceptance rate {acpt_rate:.5f}")
  finish_time = time()
  train_time_diff = finish_time - start_time

  # end of training info dump
  logging.info(f"Training time / seconds : {train_time_diff}")
  logging.info(f"Log evidence estimate : {log_evidence}")

  acpt_rate_history = jnp.array(acpt_rate_history)
  beta_history = jnp.array(beta_history)

  return samples, log_weights, log_evidence, acpt_rate_history, beta_history, train_time_diff