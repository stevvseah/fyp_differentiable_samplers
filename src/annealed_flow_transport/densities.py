"""Density functions of target distributions."""

import jax
import jax.numpy as jnp
import chex
from jax.scipy.stats import norm
import jax.scipy.linalg as slinalg
from jax.scipy.special import logsumexp
from .utils.aft_types import LogDensity

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
    chex.assert_equal_shape([self.loc, self.scale])

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
  
class ChallengingTwoDimensionalMixture(LogDensity):
  """A challenging mixture of Gaussians in two dimensions."""

  def raw_log_density(self, x: jax.Array) -> jax.Array:
    """A raw log density that we will then symmetrize.
    
    Parameters
    ----------
    x : jax.Array
      An array of particles of shape (num_particles, 2).
    
    Returns
    -------
    float
      The log density of the mixture.
    """
    mean_a = jnp.array([3.0, 0.])
    mean_b = jnp.array([-2.5, 0.])
    mean_c = jnp.array([2.0, 3.0])
    means = jnp.stack((mean_a, mean_b, mean_c), axis=0)
    cov_a = jnp.array([[0.7, 0.], [0., 0.05]])
    cov_b = jnp.array([[0.7, 0.], [0., 0.05]])
    cov_c = jnp.array([[1.0, 0.95], [0.95, 1.0]])
    covs = jnp.stack((cov_a, cov_b, cov_c), axis=0)
    log_weights = jnp.log(jnp.array([1./3, 1./3., 1./3.]))
    l = jnp.linalg.cholesky(covs)
    y = slinalg.solve_triangular(l, x[None, :] - means, lower=True, trans=0)
    mahalanobis_term = -1/2 * jnp.einsum("...i,...i->...", y, y)
    n = means.shape[-1]
    normalizing_term = -n / 2 * jnp.log(2 * jnp.pi) - jnp.log(
        l.diagonal(axis1=-2, axis2=-1)).sum(axis=1)
    individual_log_pdfs = mahalanobis_term + normalizing_term
    mixture_weighted_pdfs = individual_log_pdfs + log_weights
    return logsumexp(mixture_weighted_pdfs)

  def make_2d_invariant(self, log_density, x: jax.Array) -> jax.Array:
    density_a = log_density(x)
    density_b = log_density(jnp.flip(x))
    return jnp.logaddexp(density_a, density_b) - jnp.log(2)

  def __call__(self, samples: jax.Array) -> jax.Array:
    """Takes an array of particles as input and returns the 
    log density of each particle in an array under the 
    mixture distribution.

    Parameters
    ----------
    samples : jax.Array
      An array of particles of shape (num_particles, 2).

    Returns
    -------
    jax.Array
      An array of shape (num_particles,) of log densities of the 
      particles under Neal's funnel distribution.
    """
    chex.assert_shape(samples, (None, 2))
    
    density_func = lambda x: self.make_2d_invariant(self.raw_log_density, x)
    return jax.vmap(density_func)(samples)

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
    chex.assert_shape(samples, (None, 10))

    def unbatched_call(x):
      v = x[0]
      log_density_v = norm.logpdf(v, 0., 3.)
      log_density_other = jnp.sum(norm.logpdf(x[1:], 0., jnp.exp(v/2)))
      chex.assert_equal_shape([log_density_v, log_density_other])
      return log_density_v + log_density_other
    return jax.vmap(unbatched_call)(samples)