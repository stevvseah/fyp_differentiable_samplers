"""Density functions of target distributions."""

import jax
import jax.numpy as jnp
import chex
import numpy as np
from jax.scipy.stats import norm
import jax.scipy.linalg as slinalg
from jax.scipy.special import logsumexp
from ml_collections import ConfigDict
from .utils.aft_types import LogDensity
from .utils import cox_process_utils as cp_utils

class NormalDistribution(LogDensity):
  """Wrapper for the log density function of univariate 
  and multivariate normal distributions with diagonal 
  covariance matrices.
  
  Attributes
  ----------
  loc : jax.Array
    The mean of the distribution.
  scale : jax.Array
    The standard deviation of each component.
  """
  def __init__(self, config: ConfigDict) -> None:
    super().__init__(config)
    self.loc = jnp.array(config.loc)
    self.scale = jnp.array(config.scale)
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
  
class LogGaussianCoxPines(LogDensity):
  """Log Gaussian Cox process posterior in 2D for pine saplings data.

  This follows Heng et al 2020 https://arxiv.org/abs/1708.08396 .

  config.file_path should point to a csv file of num_points rows
  and 2 columns containing the Finnish pines data.

  config.use_whitened is a boolean specifying whether or not to use a
  reparameterization in terms of the Cholesky decomposition of the prior.
  See Section G.4 of https://arxiv.org/abs/2102.07501 for more detail.
  The experiments in the paper have this set to False.

  num_dim should be the square of the lattice sites per dimension.
  So for a 40 x 40 grid num_dim should be 1600.
  """

  def __init__(self, config: ConfigDict) -> None:
    super().__init__(config)

    # Discretization is as in Controlled Sequential Monte Carlo
    # by Heng et al 2017 https://arxiv.org/abs/1708.08396
    num_dim = config.particle_dim
    self._num_latents = num_dim
    self._num_grid_per_dim = int(jnp.sqrt(num_dim))

    bin_counts = jnp.array(
        cp_utils.get_bin_counts(self.get_pines_points(config.file_path),
                                self._num_grid_per_dim))

    self._flat_bin_counts = jnp.reshape(bin_counts, (self._num_latents))

    # This normalizes by the number of elements in the grid
    self._poisson_a = 1./self._num_latents
    # Parameters for LGCP are as estimated in Moller et al, 1998
    # "Log Gaussian Cox processes" and are also used in Heng et al.

    self._signal_variance = 1.91
    self._beta = 1./33

    self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

    def short_kernel_func(x, y):
      return cp_utils.kernel_func(x, y, self._signal_variance,
                                  self._num_grid_per_dim, self._beta)

    self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
    self._cholesky_gram = jnp.linalg.cholesky(self._gram_matrix)
    self._white_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
        2. * jnp.pi)

    half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self._cholesky_gram))))
    self._unwhitened_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
        2. * jnp.pi) - half_log_det_gram
    # The mean function is a constant with value mu_zero.
    self._mu_zero = jnp.log(126.) - 0.5*self._signal_variance

    if self.config.use_whitened:
      self._posterior_log_density = self.whitened_posterior_log_density
    else:
      self._posterior_log_density = self.unwhitened_posterior_log_density

  def  _check_constructor_inputs(self, config: ConfigDict):
    expected_members_types = [("use_whitened", bool)]
    self._check_members_types(config, expected_members_types)
    num_dim = config.particle_dim
    num_grid_per_dim = int(jnp.sqrt(num_dim))
    if num_grid_per_dim * num_grid_per_dim != num_dim:
      msg = ("num_dim needs to be a square number for LogGaussianCoxPines "
             "density.")
      raise ValueError(msg)

    if not config.file_path:
      msg = "Please specify a path in config for the Finnish pines data csv."
      raise ValueError(msg)

  def get_pines_points(self, file_path):
    """Get the pines data points."""
    with open(file_path, "rt") as input_file:
      b = np.genfromtxt(input_file, delimiter=",")
    return b

  def whitened_posterior_log_density(self, white: jax.Array) -> jax.Array:
    quadratic_term = -0.5 * jnp.sum(white**2)
    prior_log_density = self._white_gaussian_log_normalizer + quadratic_term
    latent_function = cp_utils.get_latents_from_white(white, self._mu_zero,
                                                      self._cholesky_gram)
    log_likelihood = cp_utils.poisson_process_log_likelihood(
        latent_function, self._poisson_a, self._flat_bin_counts)
    return prior_log_density + log_likelihood

  def unwhitened_posterior_log_density(self, latents: jax.Array) -> jax.Array:
    white = cp_utils.get_white_from_latents(latents, self._mu_zero,
                                            self._cholesky_gram)
    prior_log_density = -0.5 * jnp.sum(
        white * white) + self._unwhitened_gaussian_log_normalizer
    log_likelihood = cp_utils.poisson_process_log_likelihood(
        latents, self._poisson_a, self._flat_bin_counts)
    return prior_log_density + log_likelihood

  def __call__(self, x: jax.Array) -> jax.Array:
    return jax.vmap(self._posterior_log_density)(x)