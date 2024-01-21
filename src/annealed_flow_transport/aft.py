"""Implementation of the Annealed Flow Transport (AFT) sampler algorithm."""

import jax
import jax.numpy as jnp
import chex
import optax
from flax import linen as nn
from typing import Tuple, NamedTuple
from .utils.aft_types import InitialDensitySampler

class SamplesTuple(NamedTuple):
  """Container for the train, validation and test
    batches of particles.

  Attributes
  ----------
  train_samples : jax.Array
    A rank 2 array of training particles.
  validation_samples : jax.Array
    A rank 2 array of validation particles.
  test_samples : jax.Array
    A rank 2 array of test particles.
  """
  train_samples: jax.Array
  validation_samples: jax.Array
  test_samples: jax.Array


class LogWeightsTuple(NamedTuple):
  """Container for the log weights of the train, 
  validation and test batches of particles.
  
  Attributes
  ----------
  train_log_weights : jax.Array
    A rank 1 array of log weights of the training particles.
  validation_log_weights : jax.Array
    A rank 1 array of log weights of the validation particles.
  test_log_weights : jax.Array
    A rank 1 array of log weights of the test particles.
  """
  train_log_weights: jax.Array
  validation_log_weights: jax.Array
  test_log_weights: jax.Array

def initialize_particle_tuple(key: jax.Array, 
                              sampler: InitialDensitySampler, 
                              train_sampler: InitialDensitySampler
                              ) -> Tuple[SamplesTuple, LogWeightsTuple]:
  """Initialize the training, validation, and test samples, with their 
  corresponding log weights.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key.
  sampler : InitialDensitySampler
    A function to sample the test particles.
  train_sampler : InitialDensitySampler
    A function to sample the training and validation particles.

  Returns
  -------
  samples_tuple : SamplesTuple
    A named tuple containing the training, validation, and test 
    particles.
  log_weights_tuple : LogWeightsTuple
    A named tuple containing the log weights of the training, 
    validation, and test particles.
  """
  keys = jax.random.split(key, num=3)

  batch_sizes = (train_sampler.num_particles, 
                 train_sampler.num_particles, 
                 sampler.num_particles)
  samples_tuple = SamplesTuple(train_sampler(keys[0]),
                               train_sampler(keys[1]),
                               sampler(keys[2]))
  log_weights_tuple = LogWeightsTuple(*[-jnp.log(batch) * 
                                        jnp.ones(batch) for batch in batch_sizes])
  return samples_tuple, log_weights_tuple