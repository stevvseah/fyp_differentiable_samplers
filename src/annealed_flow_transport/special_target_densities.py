"""Special target densities that are not designed for all samplers to run."""

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from .utils.aft_types import LogDensityByTemp, LogDensity
from .densities import NormalDistribution

class TwoGaussians(LogDensityByTemp):
  """Special target density to test the capacity of 
  normalizing flows to learn separate mappings by 
  embedding annealing temperatures as input dimensions.  
  """
  def __init__(self, initial_log_density: LogDensity) -> None:
    self.initial_log_density = initial_log_density

    gaussian1_config = ConfigDict()
    gaussian1_config.loc = 5.
    gaussian1_config.scale = 1.

    gaussian2_config = ConfigDict()
    gaussian2_config.loc = [-3., 0]
    gaussian2_config.scale = [1., 1.]

    self.gaussian1 = NormalDistribution(gaussian1_config)
    self.gaussian2 = NormalDistribution(gaussian2_config)

  def __call__(self, beta: float, samples: jax.Array) -> jax.Array:
    """Takes an annealing temperature and an array of 
    particles, and returns the log density of each particle 
    in an array.

    For beta = 0, the log density of the initial distribution 
    is returned.

    For beta = 0.5, the log density of the first Gaussian is 
    returned.

    For beta = 1., the log density of the second Gaussian is 
    returned.
    
    Parameters
    ----------
    beta : float
      The current annealing temperature.
    samples : jax.Array
      The array of the current batch of particles.

    Returns
    -------
    jax.Array
      The log densities of the input sample as defined above.
    """
    return jnp.select([beta==0., beta==0.5, beta==1.],
                      [self.initial_log_density(samples), 
                       self.gaussian1(samples),
                       self.gaussian2(samples)],
                       default=0.)