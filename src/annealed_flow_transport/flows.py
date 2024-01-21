"""Implementation of normalizing flow models."""

import jax
import jax.numpy as jnp
import chex
from flax import linen as nn
from typing import Tuple

class DiagonalAffine(nn.Module):
  """An affine transformation with a positive definite 
  diagonal matrix.
  
  Attributes
  ----------
  particle_dim : int
    The number of dimensions of the particle positions.
  """
  particle_dim: int

  def setup(self):
    self.scale = self.param('scale', 
                            nn.initializers.zeros, 
                            (self.particle_dim,))
    self.shift = self.param('shift',
                            nn.initializers.zeros,
                            (self.particle_dim,))
  
  def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Applies the diagonal affine flow.
    
    Parameters
    ----------
    x : jax.Array
      The array of particles. The dimension of the particles 
      must match the particle_dim given to the module on 
      initialization.
    
    Returns
    -------
    x : jax.Array
      The array of transported particles.
    log_abs_det_jac : jax.Array
      An array of shape (num_particles,) that contains the 
      log of the absolute determinant jacobian of the 
      transformation in this flow, which is equivalent to 
      the sum of the scale parameters in this module.
    """
    chex.assert_shape(x, (None, self.particle_dim))
    x = jnp.exp(self.scale) * x + self.shift
    log_abs_det_jac = jnp.sum(self.scale) * jnp.ones(x.shape[0])
    return x, log_abs_det_jac