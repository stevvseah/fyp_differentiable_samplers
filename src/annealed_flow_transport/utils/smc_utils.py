"""Shared functions for SMC algorithm and variants."""

import jax
import jax.numpy as jnp
import chex
from jax.scipy.special import logsumexp
from .aft_types import LogDensityByTemp, LogDensity
from .hmc import HMCKernel
from typing import Any, Tuple, Callable

def log_effective_sample_size(log_weights: jax.Array) -> jax.Array:
	"""Computes the log effective sample size.

	ESS := (sum_i weight_i)^2 / (sum_i weight_i^2)
	log ESS = 2 log sum_i (exp log weight_i) - log sum_i (exp 2 log weight_i )

	Parameters
	----------
	log_weights : jax.Array
		Array of shape (num_particles,) containing the log weights of 
		all particles.

	Returns
	-------
	log_ess : float
		The scalar log ESS.
	"""
	chex.assert_rank(log_weights, 1)
	log_ess = 2*logsumexp(log_weights) - logsumexp(2*log_weights)
	chex.assert_rank(log_ess, 0)
	return log_ess

def resample(key: jax.Array, log_weights: jax.Array, 
						 samples: jax.Array) -> Tuple[jax.Array, jax.Array]:
	"""Simple/multinomial resampling of sample particles.

	New samples are drawn from Multinomial(softmax(log_weights), samples), 
	and new log weights returned are all equal to log(1/num_particles).

	Parameters
	----------
	key : jax.Array
		Jax PRNG key to randomly draw new samples.
	log_weights : jax.Array
		Array of shape (num_particles,) containing the log weights of
		all particles.
	samples : jax.Array
		Array of shape (num_particles, particle_dim) containing the positions
		of all particles.

	Returns
	-------
	new_samples : jax.Array
		Array of shape (num_particles, particle_dim) containing the positions
		of the resampled particles.
	new_log_weights : jax.Array
		Array of shape (num_particles,) containing the log weights of the
		resampled particles.
	"""
	chex.assert_rank(log_weights, 1)
	chex.assert_rank(samples, 2)

	num_particles = log_weights.shape[0]
	indices = jax.random.categorical(key, log_weights, shape=(num_particles,))
	new_samples = samples[indices]
	new_log_weights = -jnp.log(num_particles) * jnp.ones(num_particles)
  
	chex.assert_equal_shape([samples, new_samples])
	chex.assert_equal_shape([log_weights, new_log_weights])

	return new_samples, new_log_weights

def conditional_resample(key: jax.Array, log_weights: jax.Array, samples: jax.Array, 
												 resample_threshold: float) -> Tuple[jax.Array, jax.Array]:
	"""Performs resample if ESS is below resample_threshold * num_particles.

	I.e. resample_threshold * num_particles is the threshold ESS to trigger resampling.

	Parameters
	----------
	key : jax.Array
		Jax PRNG key to randomly draw new samples if resampling is triggered.
	log_weights : jax.Array
		Array of shape (num_particles,) containing the log weights of
		all particles.
	samples : jax.Array
		Array of shape (num_particles, particle_dim) containing the positions
		of all particles.

	Returns
	-------
	new_samples : jax.Array
		Array of shape (num_particles, particle_dim) containing the positions
		of the (possibly) resampled particles.
	new_log_weights : jax.Array
		Array of shape (num_particles,) containing the log weights of the
		(possibly) resampled particles.
	"""
	no_resample = lambda key, log_weights, samples: (samples, log_weights)
	threshold_ess = resample_threshold * log_weights.shape[0]
	log_ess = log_effective_sample_size(log_weights)
	new_samples, new_log_weights = jax.lax.cond(log_ess < jnp.log(threshold_ess), resample, 
											 no_resample, key, log_weights, samples)
	
	return new_samples, new_log_weights

class GeometricAnnealingSchedule:
  """Computes a geometric annealing schedule between initial and target log densities.
  
  Attributes
  ----------
  initial_log_density : LogDensity
    A function that takes an array of shape (num_particles, particle_dim) containing 
    the particle positions and returns an array of shape (num_particles,) containing 
    the log densities of each particle under the initial distribution.
  final_log_density : LogDensity
    A function that takes an array of shape (num_particles, particle_dim) containing 
    the particle positions and returns an array of shape (num_particles,) containing 
    the log densities of each particle under the final distribution.
  num_temps : int
    The total number of temperatures the current annealing algorithm will use.
  """
  def __init__(self, initial_log_density: LogDensity,
               final_log_density: LogDensity, 
               num_temps: int) -> None:
      self.initial_log_density = initial_log_density
      self.final_log_density = final_log_density
      self.num_temps = num_temps

  def get_beta(self, step: int) -> float:
    """Retrieves the value of the temperature at the current step.
    
    Parameters
    ----------
    step : int
      The iteration time step.
			
		Returns
		-------
		beta : float
      The annealing temperature.
    """
    beta = step / (self.num_temps-1)
    return beta
	
  def __call__(self, step: int, samples: jax.Array) -> jax.Array:
    """Computes the unnormalized interpolated density at the temperature
    of the current step.

    Parameters
    ----------
    step : int
      The iteration time step.
    samples : jax.Array
      An array of shape (num_particles, particle_dim) containing the 
      position of the particles at the current iteration.
    
    Returns
    -------
    interpolated_densities : jax.Array
      An array of the unnormalized interpolated densities at the current
      temperature.
    """
    log_densities_initial = self.initial_log_density(samples)
    log_densities_final = self.final_log_density(samples)
    beta = self.get_beta(step)
    interpolated_densities = (1-beta)*log_densities_initial + beta*log_densities_final
    return interpolated_densities

def get_log_weight_increment_no_flow(samples: jax.Array, 
                                     log_density: LogDensityByTemp, 
                                     beta: float, beta_prev: float) -> jax.Array:
  """Get the unnormalized log importance weights for the current temperature
  for an smc algorithm with no flow.
  
  Parameters
  ----------
  samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    position of the particles at the current iteration.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  beta : float
    The temperature at the current iteration.
  beta_prev : float
    The temperature at the previous iteration.
  
  Returns
  -------
  log_weight_increment : jax.Array
    An array of shape (num_particles,) containing the unnormalized 
    importance weights of the particles for the current temperature.
  """
  log_densities_current = log_density(beta, samples)
  log_densities_prev = log_density(beta_prev, samples)
  chex.assert_equal_shape([log_densities_current, log_densities_prev])
  log_weight_increment = log_densities_current - log_densities_prev
  return log_weight_increment

def get_log_weight_increment_with_flow(samples: jax.Array, 
                                       flow_apply: Callable[[dict, jax.Array], 
                                                            Tuple[jax.Array, jax.Array]], 
                                       flow_params: dict, 
                                       log_density: LogDensityByTemp, 
                                       beta: float, 
                                       beta_prev: float) -> tuple[jax.Array, jax.Array]:
  """Get the unnormalized log importance weights for the current temperature
  for an smc algorithm with flow.
  
  Parameters
  ----------
  samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    position of the particles at the current iteration.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  beta : float
    The temperature at the current iteration.
  beta_prev : float
    The temperature at the previous iteration.
  
  Returns
  -------
  log_weight_increment : jax.Array
    An array of shape (num_particles,) containing the unnormalized 
    importance weights of the particles for the current temperature.
  transported_samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    positions of the transported particles.
  """
  transported_samples, log_det_jacs = flow_apply(flow_params, samples)
  chex.assert_trees_all_equal_shapes(transported_samples, samples)
  log_densities_current = log_density(beta, transported_samples)
  log_densities_prev = log_density(beta_prev, samples)
  chex.assert_equal_shape([log_densities_current, log_densities_prev])
  chex.assert_equal_shape([log_densities_prev, log_det_jacs])
  log_weight_increment = log_densities_current - log_densities_prev + log_det_jacs
  return log_weight_increment, transported_samples

def estimate_free_energy(samples: jax.Array, 
                         log_weights: jax.Array, 
                         flow_apply: Callable[[dict, jax.Array], 
                                              Tuple[jax.Array, jax.Array]], 
                         flow_params: dict, 
                         log_density: LogDensityByTemp, 
                         beta: float, beta_prev: float, 
                         embed_time: bool = False) -> float:
  """Compute an estimate of the variational free energy. This is the 
  loss function for AFT and CRAFT.
  
  This function takes the negative weighted average of the log weight 
  increments, which is equivalent to estimating the KL divergence 
  between the distribution of the transported particles and the current 
  bridging distribution.

  Parameters
  ----------
  samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    position of the particles at the current iteration.
  log_weights : jax.Array
    An array of shape (num_particles,) containing the normalized log 
    weights of the particles in samples.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  beta : float
    The temperature at the current iteration.
  beta_prev : float
    The temperature at the previous iteration.
  embed_time : bool = False
    A boolean that indicates whether to share parameters across 
    the temperatures and embed the annealing temperature into 
    the flow.

  Returns
  -------
  div : float
    An estimate of the KL divergence between the distribution of the 
    transported particles and the current bridging distribution.
  """
  if embed_time:
     flow_apply = jax.tree_util.Partial(flow_apply, beta=beta, 
                                        beta_prev=beta_prev)

  log_weight_increment, _ = get_log_weight_increment_with_flow(samples, flow_apply, 
                                                               flow_params, log_density, 
                                                               beta, beta_prev)
  chex.assert_equal_shape([log_weight_increment, log_weights])
  div = jnp.sum(jax.nn.softmax(log_weights) * -log_weight_increment)
  return div

def estimate_free_energy_with_time_embedding(samples: jax.Array, 
                                             log_weights: jax.Array, 
                                             flow_apply: Callable[[dict, jax.Array, float, float], 
                                                                   Tuple[jax.Array, jax.Array]], 
                                             flow_params: dict, 
                                             log_density: LogDensityByTemp, 
                                             beta: float, beta_prev: float) -> float:
  """Adjusts estimate_free_energy for time-embedded samplers.

  Parameters
  ----------
  samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    position of the particles at the current iteration.
  log_weights : jax.Array
    An array of shape (num_particles,) containing the normalized log 
    weights of the particles in samples.
  flow_apply : Callable[[dict, jax.Array, float, float], 
                        Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params, samples, the current 
    annealing temperature and the previous annealing temperature to 
    transport the input samples by the underlying flow model.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  beta : float
    The temperature at the current iteration.
  beta_prev : float
    The temperature at the previous iteration.
  
  Returns
  -------
  div : float
    An estimate of the KL divergence between the distribution of the 
    transported particles and the current bridging distribution.
  """
  flow_apply_partial = jax.tree_util.Partial(flow_apply, beta=beta, 
                                             beta_prev=beta_prev)
  div = estimate_free_energy(samples, log_weights, flow_apply_partial, 
                             flow_params, log_density, beta, beta_prev)
  return div

def reweight_no_flow(log_weight_increment: jax.Array, 
                     log_weights: jax.Array) -> Tuple[jax.Array, float]:
  """Compute the new weights and log evidence increment for this temperature.

  This is for SMC algorithms that do not have a flow component.

  Parameters
  ----------
  log_weight_increment : jax.Array
    An array of shape (num_particles,) containing the unnormalized 
    importance weights of the particles for the current temperature.
  log_weights : jax.Array
    An array of shape (num_particles,) containing the normalized log 
    weights of the particles in samples.
  
  Returns
  -------
  new_log_weights : jax.Array
    An array of shape (num_particles,) containing the updated normalized 
    log weights of the particles.
  log_evidence_increment : float
    The estimate of log Z_t - log Z_{t-1}, where Z_t is the normalizing 
    constant of the t'th bridging distribution.
  """
  normalized_log_weights = jax.nn.log_softmax(log_weights)
  new_unnormalized_log_weights = normalized_log_weights + log_weight_increment
  log_evidence_increment = logsumexp(new_unnormalized_log_weights)
  new_log_weights = jax.nn.log_softmax(new_unnormalized_log_weights)
  chex.assert_equal_shape([new_log_weights, log_weights])
  chex.assert_rank(log_evidence_increment, 0)
  return new_log_weights, log_evidence_increment

def reweight_with_flow(samples: jax.Array, 
                       log_weights: jax.Array, 
                       flow_apply: Callable[[dict, jax.Array], 
                                            Tuple[jax.Array, jax.Array]], 
                       flow_params: dict, 
                       log_density: LogDensityByTemp, 
                       beta: float, beta_prev: float) -> Tuple[jax.Array, float, jax.Array]:
  """Compute the new weights and log evidence increment for this temperature.

  This is for SMC algorithms with a flow component.

  Parameters
  ----------
  samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    position of the particles at the current iteration.
  log_weights : jax.Array
    An array of shape (num_particles,) containing the normalized log 
    weights of the particles in samples.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  beta : float
    The temperature at the current iteration.
  beta_prev : float
    The temperature at the previous iteration.
  
  Returns
  -------
  new_log_weights : jax.Array
    An array of shape (num_particles,) containing the updated normalized 
    log weights of the particles.
  log_evidence_increment : float
    The estimate of log Z_t - log Z_{t-1}, where Z_t is the normalizing 
    constant of the t'th bridging distribution.
  transported_samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    positions of the transported particles.
  """
  log_weight_increment, transported_samples = get_log_weight_increment_with_flow(samples, 
                                                                                 flow_apply, 
                                                                                 flow_params, 
                                                                                 log_density, 
                                                                                 beta,
                                                                                 beta_prev)
  new_log_weights, log_evidence_increment = reweight_no_flow(log_weight_increment, log_weights)
  return new_log_weights, log_evidence_increment, transported_samples

def update_step_no_flow(key: jax.Array, samples: jax.Array, log_weights: jax.Array, 
                        log_density: LogDensityByTemp, beta: float, beta_prev: float, 
                        kernel: HMCKernel, threshold: float, step: int
                        ) -> Tuple[jax.Array, jax.Array, float, float]:
  """Produce updated samples and log weights for an SMC algorithm with no flow component.
  
  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  samples : jax.Array
    The array of particles.
  log_weights : jax.Array
    The array of log weights of the batch of particles in samples.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  beta : float
    The current annealing temperature.
  beta_prev : float
    The annealing temperature of the previous iteration.
  kernel : HMCKernel
    The HMC Kernel to be applied in the MCMC step.
  threshold : float
    The ESS threshold to trigger resampling.
  step : int
    The current iteration time step, to input into kernel in case it uses 
    a StepSizeSchedule.

  Returns
  -------
  mcmc_samples : jax.Array
    The array of particles after the importance, resampling, and mcmc steps.
  resampled_log_weights : jax.Array
    The array of log weights after the transport and resampling steps.
  log_evidence_increment : float
    The estimate of log Z_t - log Z_{t-1}, where Z_t is the normalizing 
    constant of the t'th bridging distribution.
  acceptance_rate : float
    Average acceptance rate of all HMC moves in this batch of particles.
  """
  key1, key2 = jax.random.split(key)

  # weight update
  log_weight_increment = get_log_weight_increment_no_flow(samples, log_density, beta, beta_prev)
  new_log_weights, log_evidence_increment = reweight_no_flow(log_weight_increment, log_weights)
  chex.assert_equal_shape([new_log_weights, log_weights])

  # resampling
  resampled_samples, resampled_log_weights = conditional_resample(key1, new_log_weights, 
                                                                  samples, threshold)
  chex.assert_trees_all_equal_shapes(resampled_samples, samples)
  chex.assert_equal_shape([resampled_log_weights, new_log_weights])

  # applying HMC kernel
  mcmc_samples, acceptance_rate = kernel(key2, resampled_samples, beta, step)

  return mcmc_samples, resampled_log_weights, log_evidence_increment, acceptance_rate

def update_step_with_flow(key: jax.Array, samples: jax.Array, log_weights: jax.Array, 
                          flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]], 
                          flow_params: dict, log_density: LogDensityByTemp, beta: float, 
                          beta_prev: float, kernel: HMCKernel, threshold: float, step: int
                          ) -> Tuple[jax.Array, jax.Array, float, float]:
  """Produce updated samples and log weights assuming the flow has been learned.
  
  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  samples : jax.Array
    The array of particles.
  log_weights : jax.Array
    The array of log weights of the batch of particles in samples.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]]
    A function that applies the flow.
  flow_params : dict
    The parameters of the flow.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  beta : float
    The current annealing temperature.
  beta_prev : float
    The annealing temperature of the previous iteration.
  kernel : HMCKernel
    The HMC Kernel to be applied in the MCMC step.
  threshold : float
    The ESS threshold to trigger resampling.
  step : int
    The current iteration time step, to input into kernel in case it uses 
    a StepSizeSchedule.

  Returns
  -------
  mcmc_samples : jax.Array
    The array of particles after the transport, resampling, and mcmc steps.
  resampled_log_weights : jax.Array
    The array of log weights after the transport and resampling steps.
  log_evidence_increment : float
    The estimate of log Z_t - log Z_{t-1}, where Z_t is the normalizing 
    constant of the t'th bridging distribution.
  acceptance_rate : float
    Average acceptance rate of all HMC moves in this batch of particles.
  """
  key1, key2 = jax.random.split(key)

  # flow transport and weight update
  new_log_weights, log_evidence_increment, transported_samples = reweight_with_flow(samples, 
                                                                                    log_weights, 
                                                                                    flow_apply, 
                                                                                    flow_params, 
                                                                                    log_density, 
                                                                                    beta, beta_prev)
  chex.assert_trees_all_equal_shapes(transported_samples, samples)
  chex.assert_equal_shape([new_log_weights, log_weights])

  # resampling
  resampled_samples, resampled_log_weights = conditional_resample(key1, new_log_weights, 
                                                                  transported_samples, threshold)
  chex.assert_trees_all_equal_shapes(resampled_samples, transported_samples)
  chex.assert_equal_shape([resampled_log_weights, new_log_weights])

  # applying HMC kernel
  mcmc_samples, acceptance_rate = kernel(key2, resampled_samples, beta, step)

  return mcmc_samples, resampled_log_weights, log_evidence_increment, acceptance_rate

def log_conditional_ess(log_weights: jax.Array, 
                        log_weight_increment: jax.Array) -> float:
  """Computes the log conditional effective sample size as defined in 
  Zhou, Y., Johansen, A. M., & Aston, J. A. (2016) 
  (https://www.tandfonline.com/doi/full/10.1080/10618600.2015.1060885)

  Parameters
  ----------
  log_weights : jax.Array
    The normalized log weights of a batch of particles.
  log_weight_increment : jax.Array
    An array of shape (num_particles,) containing the unnormalized 
    importance weights of the particles for the current temperature.
  
  Returns
  -------
  log_cess : float
    The log conditional effective sample size.
  """
  chex.assert_rank(log_weights, 1)
  chex.assert_equal_shape([log_weights, log_weight_increment])
  num_particles = log_weights.shape[0]
  log_numerator = jnp.log(num_particles) + 2*logsumexp(log_weights + log_weight_increment)
  log_denominator = logsumexp( log_weights + 2*log_weight_increment )
  log_cess = log_numerator - log_denominator
  return log_cess

def bisection_step(boundaries: Tuple[float, float], unused_input: None, 
                   eval_func: Callable) -> Tuple[Tuple[float, float], None]:
  """A single step of the bisection method, designed to be 
  jax scanned.

  This implementation of bisection step assumes eval_func 
  to be 

  Parameters
  ----------
  boundaries : Tuple[float, float]
    The left and right boundaries to search for the zero of 
    eval_func.
  unused_input : None
    Filler input to match the requirements of jax.lax.scan.
  eval_func : Callable
    The function to find the zero of.

  Returns
  -------
  Tuple[float, float]
    The new boundaries after one step of the bisection method.
  None
    Filler output to match the requirements of jax.lax.scan.
  """
  a, b = boundaries
  m = 0.5*(a + b)
  a_new = jax.lax.cond(eval_func(b)*eval_func(m) <= 0., lambda x: m, lambda x: a, a)
  b_new = jax.lax.cond(eval_func(a)*eval_func(m) <= 0., lambda y: m, lambda y: b, b)
  a_new, b_new = jax.lax.cond(eval_func(a)*eval_func(b) > 0, lambda x, y: (b, b), 
                              lambda x, y: (x, y), a_new, b_new)
  a_new, b_new = jax.lax.cond(eval_func(m) == 0., lambda x, y: (m, m), 
                              lambda x, y: (x, y), a_new, b_new)
  return (a_new, b_new), None

def adaptive_temp_search(samples: jax.Array, log_weights: jax.Array, 
                         beta: float, log_density: LogDensityByTemp, 
                         num_search_iters: int, threshold: float) -> float:
  """Searches for the next annealing temperature by doing a bisection method 
  search for the value of beta_next that produces a conditional ESS equal to 
  the input threshold value multiplied by the number of particles.

  Parameters
  ----------
  samples : jax.Array
    An array storing the current positions of a batch of particles.
  log_weights : jax.Array
    The log weights of the particles in samples.
  beta : float
    The current annealing temperature.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  num_search_iter : int
    The number of bisection steps to iterate through in the search.
  threshold : float
    Ratio of the target CESS to the number of particles in the batch.

  Returns
  -------
  beta_next : float
    The next annealing temperature.
  """
  chex.assert_equal(samples.shape[0], log_weights.shape[0])
  num_particles = log_weights.shape[0]
  
  def eval_func(x):
    log_weight_increment = get_log_weight_increment_no_flow(samples, log_density, x, beta)
    return log_conditional_ess(log_weights, log_weight_increment) - jnp.log(threshold*num_particles)

  bisection_step_partial = jax.tree_util.Partial(bisection_step, eval_func=eval_func)
  (a, b), _ = jax.lax.scan(bisection_step_partial, (beta, 1.), None, num_search_iters)
  beta_next = 0.5*(a + b)
  return beta_next

def adaptive_temp_search_with_flow(samples: jax.Array, 
                                   flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]], 
                                   flow_params: dict, log_weights: jax.Array, 
                                   beta: float, log_density: LogDensityByTemp, 
                                   num_search_iters: int, threshold: float) -> float:
  """Searches for the next annealing temperature by doing a bisection method 
  search for the value of beta_next that produces a conditional ESS equal to 
  the input threshold value multiplied by the number of particles.

  Parameters
  ----------
  samples : jax.Array
    An array storing the current positions of a batch of particles.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]]
    A function that applies the flow.
  flow_params : dict
    The parameters of the flow.
  log_weights : jax.Array
    The log weights of the particles in samples.
  beta : float
    The current annealing temperature.
  log_density : LogDensityByTemp
    A function taking as input a temperature and the array of particles, 
    returning the unnormalized density of the particles under the bridging 
    distribution at the input temperature.
  num_search_iter : int
    The number of bisection steps to iterate through in the search.
  threshold : float
    Ratio of the target CESS to the number of particles in the batch.

  Returns
  -------
  beta_next : float
    The next annealing temperature.
  """
  chex.assert_equal(samples.shape[0], log_weights.shape[0])
  num_particles = log_weights.shape[0]
  
  def eval_func(x):
    log_weight_increment, _ = get_log_weight_increment_with_flow(samples, flow_apply, flow_params, 
                                                                 log_density, x, beta)
    return log_conditional_ess(log_weights, log_weight_increment) - jnp.log(threshold*num_particles)

  bisection_step_partial = jax.tree_util.Partial(bisection_step, eval_func=eval_func)
  (_, beta_next), _ = jax.lax.scan(bisection_step_partial, (beta, 1.), None, num_search_iters)
  return beta_next