"""Train a flow model by MLE. The reverse of the flow must be implemented."""

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from time import time
from absl import logging
from typing import Tuple
from . import flows
import optax
from .samplers import NormalSampler
from .densities import NormalDistribution
from .train import get_optimizer, value_or_none

def train(config: ConfigDict
          ) -> Tuple[jax.Array, float, dict, optax.OptState]:
  """Trains a flow model by MLE and returns a sample from 
  the initial distribution transported by the finalized 
  flow model.

  Parameters
  ----------
  config : ConfigDict
    The config dict specifying the configurations of the 
    training and algorithm.
  
  Returns
  -------
  transformed_samples : jax.Array
    Samples transported by the trained flow model.
  full_loss : float
    The final loss obtained from the training phase.
  params : dict
    The parameters of the flow after training.
  opt_state : optax.OptState
    The optimization state of the flow after training.
  """

  data_file_path = config.data_file_path
  with open(data_file_path, 'r') as f:
    data = np.genfromtxt(f, delimiter=',')
  train_size = data.shape[0]

  key = jax.random.key(config.seed)
  key, key_ = jax.random.split(key)

  num_particles = config.num_particles
  particle_dim = config.particle_dim

  # initial distribution config
  loc = config.initial_density_config.loc
  scale = config.initial_density_config.scale
  initial_log_density = NormalDistribution(config.initial_density_config)
  sampler = NormalSampler(num_particles, particle_dim, loc, scale)

  # flow config
  flow = getattr(flows, config.flow_config.type)(config)
  params = flow.init(key_, data, False)

  def loss_fn(params, x):
    z, ildj = flow.apply(params, x, False)
    loss = jnp.mean(-initial_log_density(z) - ildj)
    return loss
  loss_val_and_grad = jax.jit( jax.value_and_grad(loss_fn) )

  # optimizer config
  initial_learning_rate = config.initial_learning_rate
  boundaries_and_scales = value_or_none('boundaries_and_scales', config)
  opt = get_optimizer(initial_learning_rate, boundaries_and_scales)
  opt_state = opt.init(params)

  # training config
  num_train_iters = config.num_train_iters
  batch_size = config.train_batch_size
  report_interval = config.report_interval

  # training loop
  logging.info('Launching training...')
  time_start = time()
  for k in range(num_train_iters):
    
    # create a batch
    idx = jnp.arange(k*batch_size, (k+1)*batch_size) % train_size
    samples = data[idx]

    # loss and grad
    loss, grad = loss_val_and_grad(params, samples)
    time_elapsed = time() - time_start

    if not k%report_interval:
      nb_epoch = float(k+1)*batch_size / train_size
      full_loss, _ = loss_val_and_grad(params, data)
      logging.info(f'Time:{time_elapsed:3.1f}sec \t Epoch:{nb_epoch:2.1f} \t Loss:{full_loss:2.2f}')
  
    # update parameters
    updates, opt_state = opt.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
  
  finish_time = time()
  train_time_diff = finish_time - time_start

  # end-of-training info dump
  logging.info(f'Training time / seconds : {train_time_diff}')
  logging.info(f'Final loss : {full_loss}')

  # reversing flow
  noise = sampler(key)
  transformed_samples, _ = flow.apply(params, noise, True)

  return transformed_samples, full_loss, params, opt_state