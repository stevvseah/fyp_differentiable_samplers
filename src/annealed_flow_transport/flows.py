"""Implementation of normalizing flow models."""

import jax
import jax.numpy as jnp
import chex
from flax import linen as nn
from typing import Tuple
from ml_collections import ConfigDict

class DiagonalAffine(nn.Module):
  """An affine transformation with a positive definite 
  diagonal matrix.
  
  Attributes
  ----------
  config : ConfigDict
  The config dict related to flow configurations.
  """
  config: ConfigDict

  def setup(self):
    self.particle_dim = self.config.particle_dim
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
  
class TimeEmbeddedDiagonalAffine(nn.Module):
  """An affine transformation with a positive definite 
  diagonal matrix that embeds the current and previous 
  annealing temperature as parameters.

  Set the embed_time attribute to True and flow type to 
  DiagonalAffine in the config file to select this flow.

  Attributes:
  -----------
  config : ConfigDict
  The config dict related to flow configurations.
  """
  config: ConfigDict

  def setup(self):
    self.particle_dim = self.config.particle_dim
    self.time_embedding_dim = self.config.flow_config.time_dim
    self.scale = self.param('scale', 
                            nn.initializers.normal(), 
                            (self.particle_dim,))
    self.shift = self.param('shift', 
                            nn.initializers.normal(), 
                            (self.particle_dim,))
  
  @nn.compact
  def __call__(self, x: jax.Array, beta: float, 
               beta_prev: float) -> Tuple[jax.Array, jax.Array]:
    """Applies the diagonal affine flow with time embedding.
    
    Parameters
    ----------
    x : jax.Array
      The array of particles. The dimension of the particles 
      must match the particle_dim given to the module on 
      initialization.
    beta : float
      The current annealing temperature.
    beta_prev : float
      The previous annealing temperature.
    
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
    i = jnp.arange(self.time_embedding_dim)
    i = i.at[1::2].add(-1)
    i /= self.time_embedding_dim
    
    embd_beta = 10*beta / ( 10000**i )
    embd_beta = embd_beta.at[::2].set(jnp.sin(embd_beta[::2]))
    embd_beta = embd_beta.at[1::2].set(jnp.cos(embd_beta[1::2]))

    embd_beta_prev = 10*beta_prev / ( 10000**i )
    embd_beta_prev = embd_beta_prev.at[::2].set(jnp.sin(embd_beta_prev[::2]))
    embd_beta_prev = embd_beta_prev.at[1::2].set(jnp.cos(embd_beta_prev[1::2]))

    x_in = jnp.concatenate((self.scale, self.shift, embd_beta, embd_beta_prev))

    shift, scale = jnp.split(nn.Dense(2*self.particle_dim, 
                                      kernel_init=nn.initializers.zeros, 
                                      bias_init=nn.initializers.zeros)(x_in), 
                                      2, -1)

    chex.assert_shape(x, (None, self.particle_dim))
    x = jnp.exp(scale) * x + shift
    log_abs_det_jac = jnp.sum(scale) * jnp.ones(x.shape[0])
    return x, log_abs_det_jac
  
class AutoregressiveMLP(nn.Module):
  """An MLP constrained to have autoregressive dependency. 
  This module is not a valid flow to be selected in a sampler.

  Attributes:
  -----------
  particle_dim : int
    The dimension of an unbatched input particle.
  num_hidden_layers : int
    The number of hidden layers to be implemented.
  hidden_layer_dim : int
    The dimension of a hidden layer (for each input dim).
  out_dim : int
    The dimension of the output.
  """
  particle_dim: int
  num_hidden_layers: int
  hidden_layer_dim: int
  out_dim : int

  @nn.compact
  def __call__(self, x:jax.Array) -> jax.Array:
    """Pushes a batch of particles forward through the 
    autoregressive MLP.
    
    Parameters
    ----------
    x : jax.Array
      The array containing the batch of particles.

    Returns
    -------
    x : jax.Array
      The output batch of particles.
    """
    hid_per_dim = self.hidden_layer_dim
    prev_layer_dim = 1
    input_dim = self.particle_dim
    x = x.T

    for i in range(self.num_hidden_layers):
      layer_shape = (input_dim, prev_layer_dim, input_dim, hid_per_dim)
      weights = self.param(f'weights_{i}', nn.initializers.glorot_normal(), layer_shape)
      bias = self.param(f'bias_{i}', nn.initializers.zeros, (input_dim, hid_per_dim))
      mask = jnp.tril(jnp.ones((self.particle_dim, self.particle_dim)))
      masked_weights = mask[:, None, :, None] * weights
      x = jnp.einsum('ijkl,ij->kl', masked_weights, x) + bias
      prev_layer_dim = hid_per_dim
      x = nn.leaky_relu(x)
    
    layer_shape = (input_dim, prev_layer_dim, input_dim, self.out_dim)
    weights = self.param(f'weights_{i+1}', nn.initializers.zeros, layer_shape)
    bias = self.param(f'bias_{i+1}', nn.initializers.zeros, (input_dim, self.out_dim))
    mask = jnp.tril(jnp.ones((input_dim, input_dim)), k=-1)
    masked_weights = mask[:, None, :, None] * weights
    x = jnp.einsum('ijkl,ij->kl', masked_weights, x) + bias
    
    return x
  
class AffineInverseAutoregressiveFlow(nn.Module):
  """An inverse autoregressive flow with affine transformer.
  
  Attributes:
  -----------
  config : ConfigDict
  The config dict related to flow configurations.
  """
  config: ConfigDict

  def setup(self):
    self.autoregressiveMLP = AutoregressiveMLP(self.config.particle_dim, 
                                               self.config.flow_config.num_hidden_layers, 
                                               self.config.flow_config.hidden_layer_dim, 
                                               2)
  
  @nn.compact
  def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Applies the affine inverse autoregressive flow.
    
    Parameters
    ----------
    x : jax.Array
      The array of particles.
    
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
    z = self.autoregressiveMLP(x)
    shift = z[:, 0]
    scale = z[:, 1] + jnp.ones_like(z[:, 1])
    log_abs_det_jac = jnp.sum(jnp.log(jnp.abs(scale))) * jnp.ones(x.shape[0])
    x = scale * x + shift
    return x, log_abs_det_jac
  
class UnbatchedAffineCouplingLayer(nn.Module):
  """A coupling layer that performs an affine transformation. 
  This layer only acts on an unbatched particle, so nn.vmap is 
  expected to be called on this layer.

  This module is not a valid flow to be selected in a sampler.
  
  Attributes
  ----------
  hidden_layer_dim : int
    The dimension of the latent representation of the particle.
  num_hidden_layers : int
    The number of hidden dense layers to be applied on the particle.
  mask : jax.Array
    The coupling layer masking to be applied.
  """
  hidden_layer_dim: int
  num_hidden_layers: int
  mask: jax.Array

  def setup(self):
    self.scaling_factor = self.param('scaling_factor',
                                     nn.initializers.zeros, 
                                     (1,))
    
  @nn.compact
  def __call__(self, x: jax.Array) -> Tuple[jax.Array, float]:
    """Performs an affine transformation on a single particle.
    
    Parameters
    ----------
    x : jax.Array
      A single particle.
    
    Returns
    -------
    x : jax.Array
      The transported particles.
    log_abs_det_jac : float
      The log of the absolute determinant jacobian of the 
      transformation in this flow, which is equivalent to 
      the scale parameter in this module.
    """

    x_in = x*self.mask

    for _ in range(self.num_hidden_layers):
      x_in = nn.Dense(self.hidden_layer_dim, 
                      kernel_init=nn.initializers.glorot_normal(),
                      bias_init=nn.initializers.zeros)(x_in)
      x_in = nn.leaky_relu(x_in)

    x_in = nn.Dense(2*x.shape[0], 
                    kernel_init=nn.initializers.zeros, 
                    bias_init=nn.initializers.zeros)(x_in)
    shift, scale = jnp.split(x_in, 2, 0)

    # stabilize scale
    stabilizer = jnp.exp(self.scaling_factor)
    stabilized_scale = nn.tanh(scale/stabilizer) * stabilizer

    # mask shift and scale
    shift *= 1-self.mask
    stabilized_scale *= 1-self.mask

    x = jnp.exp(stabilized_scale) * x + shift
    log_abs_det_jac = jnp.sum(stabilized_scale)

    return x, log_abs_det_jac
  
AffineCouplingLayer = nn.vmap(UnbatchedAffineCouplingLayer, 
                              variable_axes={'params': None}, 
                              split_rngs={'params': False})

class RealNVP(nn.Module):
  """Real-valued non-volume preserving affine transformation 
  on a batch of particles.

  Attributes:
  -----------
  config : ConfigDict
  The config dict related to flow configurations.
  """
  config: ConfigDict

  def setup(self):
    self.num_coupling_layers = self.config.flow_config.num_coupling_layers
    self.hidden_layer_dim = self.config.flow_config.hidden_layer_dim
    self.num_hidden_layers = self.config.flow_config.num_hidden_layers_per_coupling
    self.particle_dim = self.config.particle_dim

  @nn.compact
  def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Applies a sequence of affine coupling layers on a 
    batch of particles.
    
    Parameters
    ----------
    x : jax.Array
      An array containing a batch of particles to be transported.

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
    mask = jnp.zeros(self.particle_dim)
    mask = mask.at[::2].set(1)

    log_abs_det_jac = 0
    for _ in range(self.num_coupling_layers):
      mask = 1-mask
      x, step_ldj = AffineCouplingLayer(self.hidden_layer_dim, 
                                   self.num_hidden_layers, 
                                   mask)(x)
      log_abs_det_jac += step_ldj
    
    return x, log_abs_det_jac
  
class UnbatchedTimeEmbeddedAffineCouplingLayer(nn.Module):
  """A coupling layer that performs an affine transformation. 
  This layer only acts on an unbatched particle, so nn.vmap is 
  expected to be called on this layer. This layer takes in the 
  current and previous annealing temperatures as input.

  This module is not a valid flow to be selected in a sampler.
  
  Attributes
  ----------
  hidden_layer_dim : int
    The dimension of the latent representation of the particle.
  num_hidden_layers : int
    The number of hidden dense layers to be applied on the particle.
  mask : jax.Array
    The coupling layer masking to be applied.
  time_embedding_dim : int
    The number of dimensions to express each of the current and 
    previous annealing temperatures as.
  """
  hidden_layer_dim: int
  num_hidden_layers: int
  mask: jax.Array
  time_embedding_dim: int

  def setup(self):
    self.scaling_factor = self.param('scaling_factor',
                                     nn.initializers.zeros, 
                                     (1,))
    
  @nn.compact
  def __call__(self, x: jax.Array, beta: float, 
               beta_prev: float) -> Tuple[jax.Array, float]:
    """Performs an affine transformation on a single particle.
    
    Parameters
    ----------
    x : jax.Array
      A single particle.
    beta : float
      The current annealing temperature.
    beta_prev : float
      The previous annealing temperature.
    
    Returns
    -------
    x : jax.Array
      The transported particles.
    log_abs_det_jac : float
      The log of the absolute determinant jacobian of the 
      transformation in this flow, which is equivalent to 
      the scale parameter in this module.
    """

    i = jnp.arange(self.time_embedding_dim)
    i = i.at[1::2].add(-1)
    i /= self.time_embedding_dim
    
    embd_beta = 10*beta / ( 10000**i )
    embd_beta = embd_beta.at[::2].set(jnp.sin(embd_beta[::2]))
    embd_beta = embd_beta.at[1::2].set(jnp.cos(embd_beta[1::2]))

    embd_beta_prev = 10*beta_prev / ( 10000**i )
    embd_beta_prev = embd_beta_prev.at[::2].set(jnp.sin(embd_beta_prev[::2]))
    embd_beta_prev = embd_beta_prev.at[1::2].set(jnp.cos(embd_beta_prev[1::2]))

    x_in = jnp.concatenate((x*self.mask, embd_beta, embd_beta_prev))

    for _ in range(self.num_hidden_layers):
      x_in = nn.Dense(self.hidden_layer_dim, 
                      kernel_init=nn.initializers.glorot_normal(),
                      bias_init=nn.initializers.zeros)(x_in)
      x_in = nn.leaky_relu(x_in)

    x_in = nn.Dense(2*x.shape[0], 
                    kernel_init=nn.initializers.zeros, 
                    bias_init=nn.initializers.zeros)(x_in)
    shift, scale = jnp.split(x_in, 2, 0)

    # stabilize scale
    stabilizer = jnp.exp(self.scaling_factor)
    stabilized_scale = nn.tanh(scale/stabilizer) * stabilizer

    # mask shift and scale
    shift *= 1-self.mask
    stabilized_scale *= 1-self.mask

    x = jnp.exp(stabilized_scale) * x + shift
    log_abs_det_jac = jnp.sum(stabilized_scale)

    return x, log_abs_det_jac
  
TimeEmbeddedAffineCouplingLayer = nn.vmap(UnbatchedTimeEmbeddedAffineCouplingLayer, 
                                          variable_axes={'params': None}, 
                                          split_rngs={'params': False}, 
                                          in_axes=(0, None, None))

class TimeEmbeddedRealNVP(nn.Module):
  """Real-valued non-volume preserving affine transformation 
  on a batch of particles, using the current and previous 
  annealing temperatures.

  Attributes:
  -----------
  config : ConfigDict
  The config dict related to flow configurations.
  """
  config: ConfigDict

  def setup(self):
    self.num_coupling_layers = self.config.flow_config.num_coupling_layers
    self.hidden_layer_dim = self.config.flow_config.hidden_layer_dim
    self.num_hidden_layers = self.config.flow_config.num_hidden_layers_per_coupling
    self.particle_dim = self.config.particle_dim
    self.time_embedding_dim = self.config.flow_config.time_dim

  @nn.compact
  def __call__(self, x: jax.Array, beta: float, 
               beta_prev: float) -> Tuple[jax.Array, jax.Array]:
    """Applies a sequence of time-embedded affine coupling 
    layers on a batch of particles.
    
    Parameters
    ----------
    x : jax.Array
      An array containing a batch of particles to be transported.
    beta : float
      The current annealing temperature.
    beta_prev : float
      The previous annealing temperature.

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
    mask = jnp.zeros(self.particle_dim)
    mask = mask.at[::2].set(1)

    log_abs_det_jac = 0
    for _ in range(self.num_coupling_layers):
      mask = 1-mask
      x, step_ldj = TimeEmbeddedAffineCouplingLayer(self.hidden_layer_dim, 
                                                    self.num_hidden_layers, 
                                                    mask, 
                                                    self.time_embedding_dim)(
                                                    x, beta, beta_prev)
      log_abs_det_jac += step_ldj
    
    return x, log_abs_det_jac
  
class UnbatchedInvertibleAffineCouplingLayer(nn.Module):
  """A coupling layer that performs an affine transformation. 
  This layer only acts on an unbatched particle, so nn.vmap is 
  expected to be called on this layer. This layer can be 
  inverted.

  This module is not a valid flow to be selected in a sampler.
  
  Attributes
  ----------
  hidden_layer_dim : int
    The dimension of the latent representation of the particle.
  num_hidden_layers : int
    The number of hidden dense layers to be applied on the particle.
  mask : jax.Array
    The coupling layer masking to be applied.
  """
  hidden_layer_dim: int
  num_hidden_layers: int
  mask: jax.Array

  def setup(self):
    self.scaling_factor = self.param('scaling_factor',
                                     nn.initializers.zeros, 
                                     (1,))
    
  @nn.compact
  def __call__(self, x: jax.Array, invert: bool
               ) -> Tuple[jax.Array, float]:
    """Performs an affine transformation on a single particle.
    
    Parameters
    ----------
    x : jax.Array
      A single particle.
    invert : bool
      A boolean to indicate whether or not to invert the flow.
    
    Returns
    -------
    x : jax.Array
      The transported particles.
    log_abs_det_jac : float
      The log of the absolute determinant jacobian of the 
      transformation in this flow, which is equivalent to 
      the scale parameter in this module.
    """

    x_in = x*self.mask

    for _ in range(self.num_hidden_layers):
      x_in = nn.Dense(self.hidden_layer_dim, 
                      kernel_init=nn.initializers.glorot_normal(),
                      bias_init=nn.initializers.zeros)(x_in)
      x_in = nn.leaky_relu(x_in)

    x_in = nn.Dense(2*x.shape[0], 
                    kernel_init=nn.initializers.zeros, 
                    bias_init=nn.initializers.zeros)(x_in)
    shift, scale = jnp.split(x_in, 2, 0)

    # stabilize scale
    stabilizer = jnp.exp(self.scaling_factor)
    stabilized_scale = nn.tanh(scale/stabilizer) * stabilizer

    # mask shift and scale
    shift *= 1-self.mask
    stabilized_scale *= 1-self.mask

    if invert:
      x = (x - shift)/jnp.exp(stabilized_scale)
      log_abs_det_jac = -jnp.sum(stabilized_scale)
    else:
      x = jnp.exp(stabilized_scale) * x + shift
      log_abs_det_jac = jnp.sum(stabilized_scale)

    return x, log_abs_det_jac
  
InvertibleAffineCouplingLayer = nn.vmap(UnbatchedInvertibleAffineCouplingLayer, 
                                        variable_axes={'params': None}, 
                                        split_rngs={'params': False}, 
                                        in_axes=(0, None))

class InvertibleRealNVP(nn.Module):
  """Real-valued non-volume preserving affine transformation 
  on a batch of particles. The reversed direction of the flow 
  is implemented.

  This flow is not intended to be used in any AFT algorithm.

  Attributes:
  -----------
  particle_dim : int
    The dimension of each particle.
  num_coupling_layers : int
    The number of coupling layers to compose.
  num_hidden_layers_per_coupling : int
    The number of hidden layers in each coupling layer.
  hidden_layer_dim : int
    The number of dimensions in each hidden layer.
  """
  config: ConfigDict

  def setup(self):

    self.particle_dim = self.config.particle_dim
    self.num_coupling_layers = self.config.flow_config.num_coupling_layers
    self.num_hidden_layers_per_coupling = self.config.flow_config.num_hidden_layers_per_coupling
    self.hidden_layer_dim = self.config.flow_config.hidden_layer_dim

    mask = jnp.zeros(self.particle_dim)
    mask = mask.at[::2].set(1)

    masks = [0.5*(1+(-1)**i) + mask*(-1)**(i+1) for i 
             in range(self.num_coupling_layers)]
    self.flows = [InvertibleAffineCouplingLayer(self.hidden_layer_dim,
                                                 self.num_hidden_layers_per_coupling, 
                                                 masks[i]) for i in 
                                                 range(self.num_coupling_layers)]
  
  @nn.compact
  def __call__(self, x: jax.Array, invert: bool = False
               ) -> Tuple[jax.Array, jax.Array]:
    """Applies a sequence of invertible affine coupling 
    layers on a batch of particles.
    
    Parameters
    ----------
    x : jax.Array
      An array containing a batch of particles to be transported.
    invert : bool = False
      A boolean to indicate whether or not to reverse the flow.

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
    if invert:
      flow_list = reversed(self.flows)
    else:
      flow_list = self.flows

    log_abs_det_jac = 0
    for flow in flow_list:
      x, step_ldj = flow(x, invert)
      log_abs_det_jac += step_ldj

    return x, log_abs_det_jac