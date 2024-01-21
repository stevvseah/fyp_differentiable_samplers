"""Sequential Monte Carlo (SMC) sampler algorithm."""

import jax
import jax.numpy as jnp
from .utils.smc_utils import update_step_no_flow
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

def smc_apply(key: jax.Array, log_density: LogDensityByTemp, 
              initial_sampler: InitialDensitySampler, 
              kernel: HMCKernel, threshold: float, 
              num_temps: int) -> Tuple[jax.Array, jax.Array, float]:
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

  Returns
  -------
  dict
    A dictionary with the keys 'x', 'w', 'e' and 'a', storing the 
    final samples, final log weights, log evidence estimate, and 
    acceptance rates of the HMC kernel respectively.
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
  (final_samples, final_log_weights, final_temp), (log_evidence_increments, acpt_rates) = scan_step(initial_state, 
                                                                                                    per_step_input)
  
  return {'x': final_samples, 'w': final_log_weights, 'e': jnp.sum(log_evidence_increments), 'a': acpt_rates}