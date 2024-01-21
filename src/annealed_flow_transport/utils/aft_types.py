"""Custom defined types."""

import jax
import jax.numpy as jnp
import optax
import ml_collections

class LogDensity:
  """Wrapper class for log density functions."""
  
  def __call__(self, samples: jax.Array) -> jax.Array:
    """Abstract method to return the log density of an 
    input array of particles, to be defined in a subclass.
    
    Parameters
    ----------
    samples : jax.Array
      An array of shape (num_particles, particle_dim) 
      containing the positions of a batch of particles.
    
    Returns
    -------
    jax.Array
      An array of shape (num_particles,) containing the 
      log densities of each particle in samples.
    """
    raise NotImplementedError

class LogDensityByTemp:
  """Container that also behaves as a function that returns the 
  interpolated density of a sample at an input temperature.

  Attributes
  ----------
  initial_log_density : LogDensity
    A function that takes in an array of particles and outputs 
    an array of log densities of the particles under the initial 
    proposal distribution.
  final_log_density : LogDensity
    A function that takes in an array of particles and outputs 
    an array of log densities of the particles under the target 
    distribution.
  """
  def __init__(self, 
               initial_log_density: LogDensity, 
               final_log_density: LogDensity) -> None:
    self.initial_log_density = initial_log_density
    self.final_log_density = final_log_density
  
  def __call__(self, beta: float, samples: jax.Array) -> jax.Array:
    """Computes the unnormalized interpolated density at the input 
    temperature.

    Parameters
    ----------
    beta : float
      The current temperature.
    samples : jax.Array
      An array of shape (num_particles, particle_dim) containing the 
      position of the particles at the current iteration.
    
    Returns
    -------
    interpolated_densities : jax.Array
      An array of the unnormalized interpolated densities at the current
      temperature.
    """
    log_density_initial = self.initial_log_density(samples)
    log_density_final = self.final_log_density(samples)
    interpolated_density = (1-beta)*log_density_initial + beta*log_density_final
    return interpolated_density
  
class StepSizeSchedule:
  """Dummy class that step size schedulers should inherit from 
  to use with HMCKernel.
  """

class InterpolatedStepSizeSchedule(StepSizeSchedule):
  """A callable object that outputs an interpolated step size.
  
  Attributes
  ----------
  interp_step_times : list
    A list of step times to interpolate along.
  interp_step_sizes : list
    A list of step sizes to interpolate. Its elements must 
    correspond to the step times in interp_step_times.
  num_temps : int
    The total number of temperatures that the annealing algorithm
    is using.
  """
  def __init__(self, interp_step_times:list, 
               interp_step_sizes:list, 
               num_temps:int) -> None:
    self.interp_step_times = interp_step_times
    self.interp_step_sizes = interp_step_sizes
    self.num_temps = num_temps

  def __call__(self, step:int) -> float:
    """Function to return an interpolated step size, given the 
    current step index.
    
    Parameters
    ----------
    step : int
      The current iteration step index.
    step_size : float
      the appropriate interpolated step size for the input step
      index.
    """
    beta = step/(self.num_temps-1)
    step_size = jnp.interp(beta, 
                           jnp.array(self.interp_step_times), 
                           jnp.array(self.interp_step_sizes))
    return step_size
  
class InitialDensitySampler:
  """Wrapper class for samplers of initial densities to standardize 
  their signatures.
  
  Attributes
  ----------
  num_particles : int
    The number of particles in this batch.
  particle_dim : int
    The dimension of the particle positions.
  """

  def __init__(self, num_particles: int, 
               particle_dim: int) -> None:
    self.num_particles = num_particles
    self.particle_dim = particle_dim

  def __call__(self, key: jax.Array) -> jax.Array:
    """Abstract method to sample from a defined distribution, 
    to be defined in a subclass.
    
    Parameters
    ----------
    key : jax.Array
      A jax PRNG key.
    
    Returns
    -------
    jax.Array
      An array of shape (num_particles, particle_dim) containing
      the particle positions as drawn from the defined initial 
      distribution.
    """
    raise NotImplementedError