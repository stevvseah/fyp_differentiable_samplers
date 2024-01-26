"""Special target densities that are not designed for all samplers to run."""

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from .utils.aft_types import LogDensityByTemp, LogDensity
from .densities import TwoNormalMixture

class DoubleMixture(LogDensityByTemp):
  """Special target density to test the capacity of 
  normalizing flows to learn separate mappings by 
  embedding annealing temperatures as input dimensions.  
  """
  def __init__(self, initial_log_density: LogDensity) -> None:
    self.initial_log_density = initial_log_density

    mixture1_config = ConfigDict()
    mixture1_config.loc1 = 5.
    mixture1_config.scale1 = 1.
    mixture1_config.loc2 = 3.
    mixture1_config.scale2 = 1.

    mixture2_config = ConfigDict()
    mixture2_config.loc1 = [-3., 0.]
    mixture2_config.scale1 = [1., 1.]
    mixture2_config.loc2 = [0., -3.]
    mixture2_config.scale2 = [1., 1.]

    self.mixture1 = TwoNormalMixture(mixture1_config)
    self.mixture2 = TwoNormalMixture(mixture2_config)

  def __call__(self, beta: float, samples: jax.Array) -> jax.Array:
    """Takes an annealing temperature and an array of 
    particles, and returns the log density of each particle 
    in an array.

    For beta = 0, the log density of the initial distribution 
    is returned.

    For beta = 0.5, the log density of the first mixture is 
    returned.

    For beta = 1., the log density of the second mixture is 
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
                       self.mixture1(samples),
                       self.mixture2(samples)],
                       default=0.)