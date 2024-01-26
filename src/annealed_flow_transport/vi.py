"""Implementation of variational inference using normalizing flows."""

import jax
import jax.numpy as jnp
import chex
import optax
from absl import logging
from time import time
from jax.scipy.special import logsumexp
from typing import Callable, Tuple
from .utils.aft_types import LogDensityByTemp, InitialDensitySampler

def get_vfe(key: jax.Array, initial_sampler: InitialDensitySampler, 
            log_density: LogDensityByTemp, beta: float, beta_prev: float,
            flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
            Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]], 
            flow_params: dict, embed_time: bool) -> Tuple[float]:
  """Generates a sample from the initial distribution, transports 
  it to the final distribution, and returns the variational free 
  energy.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  initial_sampler : InitialDensitySampler
    Callable that takes in a jax PRNG key, and outputs a sample 
    of particles under the initial distribution.
  log_density : LogDensityByTemp
    A function that takes an annealing temperature and a sample 
    of particles, and returns the log density of the sample under 
    the corresponding log density at the given temperature. In the 
    case of VI, the initial and final log densities should correspond 
    to two unique temperatures.
  beta : float
    The current annealing temperature. Use this to set log_density to 
    be the final log density function.
  beta_prev : float
    The previous annealing temperature. Use this to set log_density to 
    be the initial log density function.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
               Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]]
    A function that takes a dictionary of parameters and a sample of 
    particles, and returns a sample of transported particles and the 
    corresponding log determinant jacobian of the flow. If embed_time 
    is set to True, then this function also takes in the current and 
    previous annealing temperatures.
  flow_params : dict
    Parameters of the flow in flow_apply.
  embed_time : bool
    A boolean to indicate whether to embed time into the model as 
    another input dimension.
  
  Returns
  -------
  vfe : float
    The variational free energy, to be minimized for training a 
    flow model.
  """
  samples = initial_sampler(key)
  if embed_time:
    flow_apply = jax.tree_util.Partial(flow_apply, beta=beta, 
                                       beta_prev=beta_prev)
  transformed_samples, log_det_jacs = flow_apply(flow_params, samples)
  chex.assert_trees_all_equal_shapes(transformed_samples, samples)
  final_log_density = log_density(beta, transformed_samples)
  initial_log_density = log_density(beta_prev, samples)
  chex.assert_equal_shape([final_log_density, initial_log_density])
  vfe = jnp.mean(initial_log_density - final_log_density - log_det_jacs)
  return vfe

def get_vfe_grad_function(initial_sampler: InitialDensitySampler, 
                          log_density: LogDensityByTemp,
                          flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
                          Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]], 
                          embed_time: bool
                          ) -> Callable[[dict, jax.Array, float, float], Tuple[float, jax.Array]]:
  """Freezes the arguments of the get_vfe that remain constant 
  through the training phase, returning a simplified get_vfe 
  that also outputs the gradient with respect to the flow 
  parameters.
  
  Parameters
  ----------
  initial_sampler : InitialDensitySampler
    Callable that takes in a jax PRNG key, and outputs a sample 
    of particles under the initial distribution.
  log_density : LogDensityByTemp
    A function that takes an annealing temperature and a sample 
    of particles, and returns the log density of the sample under 
    the corresponding log density at the given temperature. In the 
    case of VI, the initial and final log densities should correspond 
    to two unique temperatures.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
               Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]]
    A function that takes a dictionary of parameters and a sample of 
    particles, and returns a sample of transported particles and the 
    corresponding log determinant jacobian of the flow. If embed_time 
    is set to True, then this function also takes in the current and 
    previous annealing temperatures.
  embed_time : bool
    A boolean to indicate whether to embed time into the model as 
    another input dimension.

  Returns
  -------
  loss_val_and_grad : Callable[[dict, jax.Array, float, float], Tuple[float, jax.Array]]
    A function that returns the variational free energy and the gradient 
    with respect to the flow parameters.
  """
  def loss_function(params, key, beta, beta_prev):
    return get_vfe(key, initial_sampler, log_density, beta, 
                   beta_prev, flow_apply, params, embed_time)
  
  loss_val_and_grad = jax.value_and_grad(loss_function)

  return loss_val_and_grad

def vi_train_step(key: jax.Array, params: dict, beta: float, beta_prev: float, 
                  opt_state: optax.OptState, opt: optax.GradientTransformation, 
                  loss_val_and_grad: Callable[[dict, jax.Array, float, float], 
                                              Tuple[float, jax.Array]]
                  ) -> Tuple[dict, optax.OptState, float]:
  """Performs one gradient descent iteration on the flow 
  parameters.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  params : dict
    The parameters of the flow model.
  beta : float
    The annealing temperature corresponding to the final 
    distribution.
  beta_prev : float
    The annealing temperature corresponding to the initial 
    distribution.
  opt_state : optax.OptState
    The current optimization state.
  opt : optax.GradientTransformation
    The optimizer of the flow model.
  loss_val_and_grad : Callable[[dict, jax.Array, float, float], 
                                Tuple[float, jax.Array]]
    A function that computes the variational free energy and 
    its gradient with respect to the flow parameters.

  Returns
  -------
  new_params : dict
    The updated flow parameters.
  new_opt_state : optax.OptState
    The updated optimization state.
  loss : float
    The variational free energy of the flow parameters 
    before the update.
  """
  loss, grad = loss_val_and_grad(params, key, beta, beta_prev)
  updates, new_opt_state = opt.update(grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  return new_params, new_opt_state, loss

def apply(key: jax.Array, params: dict, beta: float, beta_prev: float, 
          opt_state: optax.OptState, opt: optax.GradientTransformation, 
          initial_sampler: InitialDensitySampler, log_density: LogDensityByTemp,
          flow_apply: Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
          Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]], 
          embed_time: bool, num_train_iters: int, report_interval: int
          ) -> Tuple[jax.Array, jax.Array, float, jax.Array, 
                     jax.Array, dict, optax.OptState]:
  """Performs variational inference with normalizing flow.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  params : dict
    Initial parameters of the flow in flow_apply.
  beta : float
    The current annealing temperature. Use this to set log_density to 
    be the final log density function.
  beta_prev : float
    The previous annealing temperature. Use this to set log_density to 
    be the initial log density function.
  opt_state : optax.OptState
    The initial optimization state.
  opt : optax.GradientTransformation
    The optimizer for the flow parameters.
  initial_sampler : InitialDensitySampler
    Callable that takes in a jax PRNG key, and outputs a sample 
    of particles under the initial distribution.
  log_density : LogDensityByTemp
    A function that takes an annealing temperature and a sample 
    of particles, and returns the log density of the sample under 
    the corresponding log density at the given temperature. In the 
    case of VI, the initial and final log densities should correspond 
    to two unique temperatures.
  flow_apply : Callable[[dict, jax.Array], Tuple[jax.Array, jax.Array]] | 
               Callable[[dict, jax.Array, float, float], Tuple[jax.Array, jax.Array]]
    A function that takes a dictionary of parameters and a sample of 
    particles, and returns a sample of transported particles and the 
    corresponding log determinant jacobian of the flow. If embed_time 
    is set to True, then this function also takes in the current and 
    previous annealing temperatures.
  embed_time : bool
    A boolean to indicate whether to embed time into the model as 
    another input dimension.
  num_train_iters : int
    The number of iterations to train the flow for.
  report_interval : int
    The number of training iterations before reporting the training 
    status again.

  Returns
  -------
  transformed_samples : jax.Array
    An array of randomly generated particles transported from the 
    initial distribution to the final distribution by the trained 
    flow model.
  log_importance_weights : jax.Array
    Naive log importance weights computed using transformed_samples.
  log_evidence : float
    The log evidence estimate computed using transformed_samples.
  vfe_history : jax.Array
    An array recording the loss of the flow model during the 
    training phase.
  log_evidence_history : jax.Array
    An array recording the log evidence estimates produced using the 
    parameters during the training phase.
  params : dict
    The final parameters of the flow.
  opt_state : optax.OptState
    The final optimization state of the flow.
  """
  loss_val_and_grad = get_vfe_grad_function(initial_sampler, log_density, 
                                            flow_apply, embed_time)
  
  vi_step = jax.tree_util.Partial(vi_train_step, opt=opt, 
                                     loss_val_and_grad=loss_val_and_grad)
  # jit step
  logging.info('Jitting step...')
  initial_start_time = time()
  jitted_vi_step = jax.jit(vi_step)
  initial_finish_time = time()
  initial_time_diff = initial_finish_time - initial_start_time
  logging.info(f'Initial step time / seconds: {initial_time_diff}')

  # training step
  vfe_history = []
  log_evidence_history = []
  logging.info('Launching training...')
  start_time = time()
  for step in range(num_train_iters):
    key, key1, key2 = jax.random.split(key, num=3)
    params, opt_state, loss = jitted_vi_step(key1, params, beta, beta_prev, opt_state)

    # compute log evidence estimate using naive importance sampling
    samples = initial_sampler(key2)

    if embed_time:
      transformed_samples, log_det_jacs = flow_apply(params, samples, beta, beta_prev)
    else: 
      transformed_samples, log_det_jacs = flow_apply(params, samples)
    
    final_log_density = log_density(beta, transformed_samples)
    initial_log_density = log_density(beta_prev, samples)
    log_importance_weights = final_log_density - initial_log_density + log_det_jacs
    log_evidence = logsumexp(log_importance_weights) - jnp.log(initial_sampler.num_particles)

    vfe_history.append(loss)
    log_evidence_history.append(log_evidence)

    if step % report_interval == 0:
      logging.info(f'step {step:04d}: vfe {loss:.5f} \t log evidence {log_evidence:.5f}')
  finish_time = time()
  train_time_diff = finish_time - start_time

  # end-of-training info dump
  logging.info(f'Training time / seconds : {train_time_diff}')
  logging.info(f'Log evidence estimate : {log_evidence}')

  vfe_history = jnp.array(vfe_history)
  log_evidence_history = jnp.array(log_evidence_history)

  return transformed_samples, log_importance_weights, log_evidence, \
         vfe_history, log_evidence_history, params, opt_state