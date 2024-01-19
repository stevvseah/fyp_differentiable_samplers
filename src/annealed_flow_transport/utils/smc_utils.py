"""Shared functions for SMC algorithm and variants."""

import jax
import jax.numpy as jnp
import chex
from jax.scipy.special import logsumexp

def log_effective_sample_size(log_weights:jax.Array) -> jax.Array:
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

def resample(key:jax.Array, log_weights:jax.Array, 
						 samples:jax.Array) -> tuple[jax.Array, jax.Array]:
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

def conditional_resample(key:jax.Array, log_weights:jax.Array, samples:jax.Array, 
												 resample_threshold:float) -> tuple[jax.Array, jax.Array]:
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
  """Computes a geometric annealing schedule between initial and target log densities."""
	
  initial_log_density: callable[[int, jax.Array], jax.Array]
  final_log_density: callable[[int, jax.Array], jax.Array]
  num_temps: int

  def get_beta(self, step:int) -> float:
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
	
  def __call__(self, step:int, samples:jax.Array) -> jax.Array:
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

def get_log_weight_increment_no_flow(samples:jax.Array, 
                                     log_density:callable[[int, jax.Array], jax.Array], 
                                     step:int) -> jax.Array:
  """Get the unnormalized log importance weights for the current temperature
  for an smc algorithm with no flow.
  
  Parameters
  ----------
  samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    position of the particles at the current iteration.
  log_density : callable[[int, jax.Array], jax.Array]
    A function returning the unnormalized interpolated density at the 
    temperature of the current step.
  step : int
    The iteration time step.
  
  Returns
  -------
  log_weight_increment : jax.Array
    An array of shape (num_particles,) containing the unnormalized 
    importance weights of the particles for the current temperature.
  """
  log_densities_current = log_density(step, samples)
  log_densities_prev = log_density(step-1, samples)
  chex.assert_equal_shape([log_densities_current, log_densities_prev])
  log_weight_increment = log_densities_current - log_densities_prev
  return log_weight_increment

def get_log_weight_increment_with_flow(samples:jax.Array, 
                                       flow_apply:callable[[dict, jax.Array], [jax.Array, jax.Array]], 
                                       flow_params:dict, 
                                       log_density:callable[[int, jax.Array], jax.Array], 
                                       step:int) -> tuple[jax.Array, jax.Array]:
  """Get the unnormalized log importance weights for the current temperature
  for an smc algorithm with flow.
  
  Parameters
  ----------
  samples : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    position of the particles at the current iteration.
  flow_apply : callable[[dict, jax.Array], [jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  log_density : callable[[int, jax.Array], jax.Array]
    A function returning the unnormalized interpolated density at the 
    temperature of the current step.
  step : int
    The iteration time step.
  
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
  log_densities_current = log_density(step, samples)
  log_densities_prev = log_density(step-1, samples)
  chex.assert_equal_shape([log_densities_current, log_densities_prev])
  chex.assert_equal_shape([log_densities_prev, log_det_jacs])
  log_weight_increment = log_densities_current - log_densities_prev + log_det_jacs
  return log_weight_increment, transported_samples

def estimate_free_energy(samples:jax.Array, 
                         log_weights:jax.Array, 
                         flow_apply:callable[[dict, jax.Array], [jax.Array, jax.Array]], 
                         flow_params: dict, 
                         log_density: callable[[int, jax.Array], jax.Array], 
                         step:int) -> jax.Array:
  """Compute an estimate of the free energy. This is the loss function
  for AFT and CRAFT.
  
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
  flow_apply : callable[[dict, jax.Array], [jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  log_density : callable[[int, jax.Array], jax.Array]
    A function returning the unnormalized interpolated density at the 
    temperature of the current step.
  step : int
    The iteration time step.
  
  Returns
  -------
  div : float
    An estimate of the KL divergence between the distribution of the 
    transported particles and the current bridging distribution.
  """
  log_weight_increment, _ = get_log_weight_increment_with_flow(samples, flow_apply, 
                                                               flow_params, log_density, step)
  chex.assert_equal_shape([log_weight_increment, log_weights])
  div = jnp.sum(jax.nn.softmax(log_weights) * -log_weight_increment)
  return div

def reweight_no_flow(log_weight_increment:jax.Array, 
                     log_weights:jax.Array) -> tuple[jax.Array, float]:
  """Compute the new weights and log evidence increment for this temperature.

  This is for SMC algorithms that do not a flow component.

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
  new_log_weights = jax.nn.log_softmax(log_evidence_increment)
  chex.assert_equal_shape([new_log_weights, log_weights])
  chex.assert_rank(log_evidence_increment, 0)
  return new_log_weights, log_evidence_increment

def reweight_with_flow(samples:jax.Array, 
                       log_weights:jax.Array, 
                       flow_apply:callable[[dict, jax.Array], [jax.Array, jax.Array]], 
                       flow_params:dict, 
                       log_density:callable[[int, jax.Array], jax.Array], 
                       step:int) -> tuple[jax.Array, float, jax.Array]:
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
  flow_apply : callable[[dict, jax.Array], [jax.Array, jax.Array]]
    A function that takes as input flow_params and samples to transport 
    the input samples by the underlying flow model.
  flow_params : dict
    The parameters of the flow model of flow_apply.
  log_density : callable[[int, jax.Array], jax.Array]
    A function returning the unnormalized interpolated density at the 
    temperature of the current step.
  step : int
    The iteration time step.
  
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
                                                                                 step)
  new_log_weights, log_evidence_increment = reweight_no_flow(log_weight_increment, log_weights)
  return new_log_weights, log_evidence_increment, transported_samples

def update_samples_and_weights(key:jax.Array, samples:jax.Array, log_weights:jax.Array, 
                               flow_apply:callable[[dict, jax.Array], [jax.Array, jax.Array]], 
                               flow_params:dict, log_density:callable[[int, jax.Array], jax.Array], 
                               step:int, markov_kernel_apply) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """do markov kernels first"""
  pass