"""Samplers of initial distributions."""

import jax
import jax.numpy as jnp
from jax.random import normal
from .aft_types import InitialDensitySampler

class NormalSampler(InitialDensitySampler):
  """Wrapper for univariate and multivariate normal 
  distributions with diagonal covariance matrices.
  
  Attributes
  ----------
  num_particles : int
    The number of particles in this batch.
  particle_dim : int
    The dimension of the particle positions.
  loc : float|jax.Array
    The mean of the distribution.
  scale : float|jax.Array
    The standard deviation of each component.
  """
  def __init__(self, num_particles: int, particle_dim: int, 
               loc: float | list=0., 
               scale: float | list=1.) -> None:
    super().__init__(num_particles, particle_dim)
    self.loc = jnp.array(loc)
    self.scale = jnp.array(scale)
  
  def __call__(self, key: jax.Array) -> jax.Array:
    noise = normal(key, (self.num_particles, self.particle_dim))
    return self.scale * noise + self.loc