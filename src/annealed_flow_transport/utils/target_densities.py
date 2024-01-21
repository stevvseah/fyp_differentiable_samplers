"""Density functions of target distributions."""

import jax
import jax.numpy as jnp
import chex
from jax.scipy.stats import norm
from .aft_types import LogDensity

class NormalDistribution(LogDensity):
  """Wrapper for the log density function of univariate 
  and multivariate normal distributions with diagonal 
  covariance matrices.
  
  Attributes
  ----------
  loc : float | Array
    The mean of the distribution.
  scale : float | Array
    The standard deviation of each component.
  """
  def __init__(self, loc: float | list[float]=0., 
               scale: float| list[float]=1.) -> None:
    super().__init__()
    self.loc = jnp.array(loc)
    self.scale = jnp.array(scale)
    chex.assert_equal_shape([loc, scale])

  def __call__(self, samples: jax.Array) -> jax.Array:
    """Takes an array of particles as input and returns the 
    log density of each particle in an array under the defined 
    normal distribution.
    
    Parameters
    ----------
    samples : jax.Array
      An array of particles of shape (num_particles, particle_dim).
    
    Returns
    -------
    jax.Array
      An array of shape (num_particles,) of log densities of the 
      particles under the defined normal distribution.
    """
    chex.assert_rank(samples, 2)
    log_densities = norm.logpdf(samples, self.loc, self.scale)
    chex.assert_equal_shape([samples, log_densities])
    return jnp.sum(log_densities, axis=1)
  
class NealsFunnel(LogDensity):
  """Wrapper for the log density function of Neal's funnel 
  distribution."""
  
  def __call__(self, samples: jax.Array) -> jax.Array:
    """Takes an array of particles as input and returns the 
    log density of each particle in an array under Neal's 
    funnel distribution.

    Parameters
    ----------
    samples : jax.Array
      An array of particles of shape (num_particles, particle_dim).

    Returns
    -------
    jax.Array
      An array of shape (num_particles,) of log densities of the 
      particles under Neal's funnel distribution.
    """
    def unbatched_call(x):
      v = x[0]
      log_density_v = norm.logpdf(v, 0., 3.)
      log_density_other = jnp.sum(norm.logpdf(x[1:], 0., jnp.exp(v/2)))
      chex.assert_equal_shape([log_density_v, log_density_other])
      return log_density_v + log_density_other
    return jax.vmap(unbatched_call)(samples)