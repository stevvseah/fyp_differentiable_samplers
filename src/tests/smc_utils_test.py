"""Unit tests for smc_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from annealed_flow_transport.utils import smc_utils as su
import chex
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

def _assert_equal_vec(tester: parameterized.TestCase, 
                      vec1: jax.Array, vec2: jax.Array, **kwargs):
    """Check if two arrays containing floats have almost equal elements.
    
    Parameters
    ----------
    tester: parameterized.TestCase
      ABSL unit test object.
    vec1: jax.Array
      First array to compare.
    vec2: jax.Array
      Second array to compare.
    """
    tester.assertTrue(jnp.allclose(vec1, vec2, **kwargs))

class ResamplingTest(parameterized.TestCase):
  """Tests for resampling functionality of smc_utils."""

  @parameterized.parameters(
    (1/5 * jnp.ones(5), -3.7),
    (jnp.array([0.3, 0.2, 0.5]), -5.4)
  )
  def test_ess(self, weights, arbitrary_num):
    """Test if log ESS of unnormalized weights are computed correctly.

    Parameters
    ----------
    weights : jax.Array
      Normalized weights (not log weights) of particles.
    arbitrary_num : float
      Arbitrary float to offset and unnormalize the log weights.
    """
    true_log_ess = jnp.log(1./jnp.sum(weights**2))
    
    unnormalized_log_weights = jnp.log(weights) + arbitrary_num
    test_log_ess = su.log_effective_sample_size(unnormalized_log_weights)

    self.assertAlmostEqual(test_log_ess, true_log_ess, delta=1e-6)

  @parameterized.parameters(
    (1, jnp.array([0.5, 0.25, 0.25]), 10000, -5.2),
    (42, jnp.array([0.4, 0.3, 0.2, 0.1]), 10000, -1.1)
  )
  def test_resampling(self, seed, large_weights, num_particles, arbitrary_num):
    """Test if resampling correctly samples from existing particles.

    This test initializes an initial sample of shape (num_particles, 1), with 
    weights array (large_weights, 0, 0, ..., 0). This weight imbalance is designed
    to cause most of the initial sample to be resampled into the first few particles. 
    Then the resampled particles should follow the multinomial(large_weights, 
    samples[:large_weights.shape[0]]) distribution. We test the empirical mean and 
    variance of the resampled particles against the mean and variance of this 
    multinomial distribution.

    Parameters
    ----------
    seed : int
      Provides seed for generation of PRNG key used to sample initial particles and
      perform resampling.
    large_weights : jax.Array
      The non-zero normalized weights of the initial sample. Must not have more elements
      than num_particles.
    num_particles : int
      The number of particles in the initial sample.
    arbitrary_num : float
      Arbitrary float to offset and unnormalize the log weights.
    """
    self.assertLess(large_weights.shape[0], num_particles)

    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    samples = jax.random.normal(key1, (num_particles, 1))
    weights = jnp.concatenate( (large_weights, jnp.zeros(num_particles-large_weights.shape[0])) )
    unnormalized_log_weights = jnp.log(weights) + arbitrary_num
    
    # compute empirical mean and variance
    new_samples, new_log_weights = su.resample(key2, unnormalized_log_weights, samples)
    test_mean = jnp.mean(new_samples)
    test_var = jnp.var(new_samples)

    # compute the multinomial mean and variance
    true_mean = jnp.sum( large_weights * samples[:large_weights.shape[0], -1] )
    true_var = jnp.sum( large_weights * (samples[:large_weights.shape[0], -1] - true_mean)**2 )
    true_log_weights = -jnp.log(num_particles) * jnp.ones(num_particles)

    self.assertSequenceAlmostEqual(new_log_weights, true_log_weights, delta=1e-6)
    self.assertAlmostEqual(test_mean, true_mean, delta=1e-2)
    self.assertAlmostEqual(test_var, true_var, delta=1e-2)

  @parameterized.parameters(
    (1, 100, 2),
    (5, 50, 4)
  )
  def test_conditional_resampling(self, seed, num_particles, particle_dim):
    """Test if conditional resampling performs correct resampling based on
    threshold values.

    This test generates a random initial sample and log weights, and computes the 
    resampled particles based on a fixed key. The log_ess is then computed to generate 
    thresholds that will trigger and and fail to trigger the resampling scheme.
    conditional_resample is then run using the fixed key on the inital samples using 
    each of the thresholds, and the produced samples and log weights are tested for 
    equality with the initial (for no resampling) and resampled (for, well, resampled) 
    samples and log weights.

    Parameters
    ----------
    seed : int
      Provides seed for generation of PRNG key used to sample initial particles and
      perform resampling.
    num_particles : int
      The number of particles in the initial sample.
    particle_dim : int
      Dimension of each particle.    
    """
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, num=3)

    # compute the true resampled particles and log weights
    samples = jax.random.normal(key1, (num_particles, particle_dim))
    log_weights = jax.random.normal(key2, (num_particles,))
    log_ess = su.log_effective_sample_size(log_weights)
    resamples, uniform_log_weights = su.resample(key3, log_weights, samples)

    # define thresholds that should not(lower) and should(upper) trigger resampling
    threshold_lower = 0.9/num_particles*jnp.exp(log_ess)
    threshold_upper = 1.1/num_particles*jnp.exp(log_ess)

    # generate test output from conditional_resampling
    test_samples, test_log_weights = su.conditional_resample(key3, log_weights, 
                                                             samples, threshold_lower)
    test_resamples, test_uniform_log_weights = su.conditional_resample(key3, log_weights, 
                                                                       samples, threshold_upper)
    
    _assert_equal_vec(self, test_samples, samples)
    _assert_equal_vec(self, test_resamples, resamples)
    self.assertSequenceAlmostEqual(test_log_weights, log_weights)
    self.assertSequenceAlmostEqual(test_uniform_log_weights, uniform_log_weights)

class UpdatesTest(parameterized.TestCase):
  """Tests for weight update functionalities of smc_utils."""

  @parameterized.parameters(
    (1, 0.1, 0., -1., 2., 1., 1., 4., 5.),
    (5, -2., -1.5, 1., 3., 0.5, 0.5, 1., 1.2)
  )
  def test_estimate_free_energy(self, seed:int, 
                                mean_a:float, mean_b:float, mean_c:float, mean_d:float, 
                                var_a:float, var_b:float, var_c:float, var_d:float):
    """Tests if estimate_free_energy correctly computes the KL-divergence between the 
    distribution of the transported particles and the current bridging distribution.

    Let A, B, C and D be Gaussian distributions.

    This test draws particles from A, and generates log importance weights such that 
    the weighted particles approximate B. A flow transporting the particles from B to 
    C is then applied. We set the current bridging distribution to D, so that the output
    of estimate_free_energy is the KL divergence D_KL(C||D). This can be computed 
    analytically, so we check whether the estimate provided by estimate_free_energy 
    is close (within +-0.1) to the true value of the KL divergence.

    Parameters
    ----------
    seed : int
      Provides seed for generation of PRNG key used to sample initial particles.
    mean_a/b/c/d : float
      Value of the mean of Gaussian distribution A/B/C/D.
    var_a/b/c/d : float
      Value of the variance Gaussian distribution A/B/C/D.
    """
    num_particles = 100000
    key = jax.random.PRNGKey(seed)
    key, key_ = jax.random.split(key)

    samples = jnp.sqrt(var_a) * jax.random.normal(key, (num_particles, 1)) + mean_a
    log_density_a = norm.logpdf(samples, mean_a, 
                                       jnp.sqrt(var_a)).flatten()
    log_density_b = norm.logpdf(samples, mean_b, 
                                         jnp.sqrt(var_b)).flatten()
    log_weights = log_density_b - log_density_a

    def flow_apply(unused_params, samples):
      transported_samples = jnp.sqrt(var_c/var_b)*(samples-mean_b)+mean_c
      log_det_jacs = jnp.log(jnp.sqrt(var_c/var_b)) * jnp.ones(num_particles)
      return transported_samples, log_det_jacs
    
    def kl_div(mean0, var0, mean1, var1):
      return 0.5 * (
        var0 / var1 + jnp.square(mean1 - mean0) / var1 - 1. + jnp.log(var1) - 
        jnp.log(var0))
    
    def step_density(beta, x):
      log_density_b = norm.logpdf(x, mean_b, jnp.sqrt(var_b)).flatten()
      log_density_d = norm.logpdf(x, mean_d, jnp.sqrt(var_d)).flatten()
      return (1-beta)*log_density_b + beta*log_density_d
    
    test_div = su.estimate_free_energy(samples=samples, log_weights=log_weights, 
                                       flow_apply=flow_apply, flow_params=None, 
                                       log_density=step_density, beta=1, beta_prev=0)
    true_div = kl_div(mean_c, var_c, mean_d, var_d)

    self.assertAlmostEqual(test_div, true_div, delta=1e-1)

if __name__ == '__main__':
  absltest.main()