"""Implementation of the HMC Markov kernel."""

import chex
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from typing import Callable, Tuple

class InterpolatedStepSizeSchedule:
  """A callable object that outputs an interpolated step sizes.
  
  Attributes
  ----------
  interp_step_times : list
    A list of step times to interpolate along.
  interp_step_sizes : list
    A list of step sizes to interpolate. Its elements must 
    correspond to the step times in interp_step_times.
  num_temps : int
    The total number of temperatures that the annealing algorithm
    is using.
  """

  def __init__(self, interp_step_times:list, 
               interp_step_sizes:list, 
               num_temps:int):
    self.interp_step_times = interp_step_times
    self.interp_step_sizes = interp_step_sizes
    self.num_temps = num_temps

  def __call__(self, step:int) -> float:
    """Function to return an interpolated step size, given the 
    current step index.
    
    Parameters
    ----------
    step : int
      The current iteration step index.
    step_size : float
      the appropriate interpolated step size for the input step
      index.
    """
    beta = step/(self.num_temps-1)
    step_size = jnp.interp(beta, 
                           jnp.array(self.interp_step_times), 
                           jnp.array(self.interp_step_sizes))
    return step_size
  
def tree_add(tree_a:chex.ArrayTree, 
             tree_b:chex.ArrayTree) -> chex.ArrayTree:
  """Performs element-wise addition between two pytrees.

  Parameters
  ----------
  tree_a : pytree
    A pytree to perform addition with.
  tree_b : pytree
    A pytree to perform addition with tree_a.

  Returns
  -------
  res : pytree
    A pytree with elements that are equal to the addition
    of the corresponding elements in tree_a and tree_b.
  """
  chex.assert_trees_all_equal_shapes(tree_a, tree_b)
  res = jax.tree_map(lambda a, b: a+b, tree_a, tree_b)
  return res


def tree_scalar_mul(tree:chex.ArrayTree, 
                    scalar:float) -> chex.ArrayTree:
  """Multiplies all the leaves in a pytree by an input scalar.

  Parameters
  ----------
  tree : pytree
    A tree that we want to perform multiplication with.
  scalar : float
    The scalar that we want to multiply all the leaves of tree
    with.

  Returns
  -------
  res : pytree
    A pytree with elements that are equal to the tree's 
    elements multiplied by scalar.
  """
  res = jax.tree_map(lambda x: x * scalar, tree)
  chex.assert_trees_all_equal_shapes(tree, res)
  return res

def momentum_step(samples_in: jax.Array,
                  momentum_in: jax.Array,
                  step_coefficient: float,
                  epsilon: float,
                  grad_log_density:Callable[[jax.Array], jax.Array]
                  ) -> jax.Array:
  """A leapfrog integrator momentum update with variable 
  momentum step_coefficient.

  Parameters
  ----------
  samples_in : jax.Array
    An array of shape (num_particles, particle_dim) containing 
    the positions of the particles.
  momentum_in : jax.Array
    An array of shape (num_particles, particle_dim) containing
    the momentum of the particles.
  step_coefficient : float
    A Scalar which is either 0.5 to represent a half step or 1.0 
    to represent a full step.
  epsilon : float
    A Scalar representing the constant step size.
  grad_log_density : Callable[[jax.Array], jax.Array]
    A function that takes in an array of shape (num_particles, 
    particle_dim) containing the positions of the particles and 
    returning an array of the same shape containing the gradients
    of the potential energy (negative log density).

  Returns
  -------
  momentum_out : jax.Array
    Array of shape (num_particles, particle_dim) containing the
    next (half or full) step momentum.
  """
  chex.assert_rank((step_coefficient, epsilon), (0, 0))
  chex.assert_trees_all_equal_shapes(samples_in, momentum_in)
  gradient_val = grad_log_density(samples_in)
  momentum_out = tree_add(
      momentum_in, tree_scalar_mul(gradient_val, step_coefficient * epsilon))
  chex.assert_trees_all_equal_shapes(momentum_in, momentum_out)
  return momentum_out

def leapfrog_step(samples_in: jax.Array,
                  momentum_in: jax.Array,
                  step_coefficient: jax.Array,
                  epsilon: jax.Array,
                  grad_log_density:Callable[[jax.Array], jax.Array]
                  ) -> Tuple[jax.Array, jax.Array]:
  """A step of the Leapfrog iteration with variable momentum step_coefficient.

  Parameters
  ----------
  samples_in : jax.Array
    An array of shape (num_particles, particle_dim) containing 
    the positions of the particles.
  momentum_in : jax.Array
    An array of shape (num_particles, particle_dim) containing
    the momentum of the particles.
  step_coefficient : float
    A Scalar which is either 0.5 to represent a half step or 1.0 
    to represent a full step.
  epsilon : float
    A Scalar representing the constant step size.
  grad_log_density : Callable[[jax.Array], jax.Array]
    A function that takes in an array of shape (num_particles, 
    particle_dim) containing the positions of the particles and 
    returning an array of the same shape containing the gradients
    of the potential energy (negative log density).

  Returns
  -------
  samples_out : jax.Array
    Array of shape (num_particles, particle_dim)
  momentum_out : jax.Array
    Array of shape (num_particles, particle_dim) containing the
    next (half or full) step momentum.
  """
  chex.assert_rank((step_coefficient, epsilon), (0, 0))
  chex.assert_trees_all_equal_shapes(samples_in, momentum_in)
  samples_out = tree_add(samples_in, tree_scalar_mul(momentum_in, epsilon))
  momentum_out = momentum_step(samples_out, momentum_in, step_coefficient,
                               epsilon, grad_log_density)
  chex.assert_trees_all_equal_shapes(samples_in, samples_out)
  return samples_out, momentum_out

def random_normal_like_tree(key:jax.Array, tree:chex.ArrayTree):
  """Generates a tree of standard Gaussian distributed values.

  Parameters
  ----------
  key : jax.Array
    A jax PRNG key to generate the values.
  tree : Pytree
    A pytree with the desired output shape.
  
  Returns
  -------
  tree_normals : Pytree
    A pytree of standard Gaussian distributed values in the 
    shape of the input tree.
  """
  tree_struct = jax.tree_util.tree_structure(tree)
  split_keys = jax.random.split(key, tree_struct.num_leaves)
  tree_keys = jax.tree_util.tree_unflatten(tree_struct, split_keys)
  tree_normals = jax.tree_util.tree_map(
      lambda x, y: jax.random.normal(key=y, shape=x.shape), tree, tree_keys)
  return tree_normals

def hmc_step(samples_in: jax.Array,
             key: jax.Array,
             epsilon: float,
             log_density: Callable[[jax.Array], jax.Array],
             grad_log_density: Callable[[jax.Array], jax.Array],
             num_leapfrog_iters: int) -> Tuple[jax.Array, jax.Array]:
  """A single step of Hamiltonian Monte Carlo.

  Parameters
  ----------
  samples_in : jax.Array
    An array of shape (num_particles, particle_dim) containing 
    the positions of the particles.
  key : jax.Array
    A jax PRNG key.
  epsilon : float
    A Scalar representing the constant step size.
  log_density : Callable[[jax.Array], jax.Array]
    A function that takes in the array of particle positions and 
    returns the log densities of each particle in an array.
  grad_log_density : Callable[[jax.Array], jax.Array]
    A function that takes in an array of shape (num_particles, 
    particle_dim) containing the positions of the particles and 
    returning an array of the same shape containing the gradients
    of the potential energy (negative log density).
  num_leapfrog_iters : int
    The number of leapfrog iterations.

  Returns
  -------
  samples_next : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    positions of the particles at the end of this HMC step.
  step_acceptance_rate : float
    Empirical average acceptance rate of HMC moves in this batch
    of particles.
  """
  chex.assert_rank(epsilon, 0)
  samples_state = samples_in
  momentum_key, acceptance_key = jax.random.split(key)
  initial_momentum = random_normal_like_tree(momentum_key, samples_in)
  # A half momentum step.
  momentum_state = momentum_step(samples_state, initial_momentum,
                                 step_coefficient=0.5,
                                 epsilon=epsilon,
                                 grad_log_density=grad_log_density)
  def scan_step(passed_state, unused_input):
    pos, mom = passed_state
    new_pos, new_mom = leapfrog_step(pos, mom, step_coefficient=1.0,
                                     epsilon=epsilon,
                                     grad_log_density=grad_log_density)
    return (new_pos, new_mom), None

  state_in = (samples_state, momentum_state)
  scan_length = num_leapfrog_iters - 1
  # (num_leapfrog_iters - 1) whole position and momentum steps.
  new_state, _ = jax.lax.scan(
      scan_step, state_in, [None] * scan_length, length=scan_length)
  samples_state, momentum_state = new_state

  # A whole position step and half momentum step.
  samples_state, momentum_state = leapfrog_step(
      samples_state,
      momentum_state,
      step_coefficient=0.5,
      epsilon=epsilon,
      grad_log_density=grad_log_density)

  # We don't negate the momentum here because it has no effect.
  # This would be required if momentum was used other than for just the energy.

  # Decide if we accept the proposed update using Metropolis correction.
  def get_combined_log_densities(pos, mom):
    pos_log_densities = log_density(pos)
    def leaf_log_density(x):
      summation_axes = tuple(range(1, len(jnp.shape(x))))
      return -0.5 * jnp.sum(jnp.square(x), axis=summation_axes)
    per_leaf_mom_log_densities = jax.tree_util.tree_map(leaf_log_density,
                                                        mom)
    mom_log_densities = jax.tree_util.tree_reduce(
        jnp.add, per_leaf_mom_log_densities)
    chex.assert_equal_shape((pos_log_densities, mom_log_densities))
    return pos_log_densities + mom_log_densities

  current_log_densities = get_combined_log_densities(samples_in,
                                                     initial_momentum)
  proposed_log_densities = get_combined_log_densities(samples_state,
                                                      momentum_state)
  num_batch = jnp.shape(current_log_densities)[0]
  exponential_rvs = jax.random.exponential(key=acceptance_key,
                                           shape=(num_batch,))

  delta_log_prob = proposed_log_densities - current_log_densities
  chex.assert_shape(delta_log_prob, (num_batch,))
  is_accepted = jnp.greater(delta_log_prob, -1.*exponential_rvs)
  chex.assert_shape(is_accepted, (num_batch,))
  step_acceptance_rate = jnp.mean(is_accepted * 1.)
  def acceptance(a, b):
    broadcast_axes = tuple(range(1, len(a.shape)))
    broadcast_is_accepted = jnp.expand_dims(is_accepted,
                                            axis=broadcast_axes)
    return jnp.where(broadcast_is_accepted, a, b)
  samples_next = jax.tree_util.tree_map(acceptance,
                                        samples_state,
                                        samples_in)
  return samples_next, step_acceptance_rate

def hmc(samples_in: jax.Array,
        key: jax.Array,
        epsilon: float,
        log_density: Callable[[jax.Array], jax.Array],
        grad_log_density: Callable[[jax.Array], jax.Array],
        num_leapfrog_iters: int,
        num_hmc_iters: int) -> Tuple[jax.Array, jax.Array]:
  """Markov kernel stacking HMC steps together.

  Parameters
  ----------
  samples_in : jax.Array
    An array of shape (num_particles, particle_dim) containing 
    the positions of the particles.
  key : jax.Array
    A jax PRNG key.
  epsilon : float
    A Scalar representing the constant step size.
  log_density : Callable[[jax.Array], jax.Array]
    A function that takes in the array of particle positions and 
    returns the log densities of each particle in an array.
  grad_log_density : Callable[[jax.Array], jax.Array]
    A function that takes in an array of shape (num_particles, 
    particle_dim) containing the positions of the particles and 
    returning an array of the same shape containing the gradients
    of the potential energy (negative log density).
  num_leapfrog_iters : int
    The number of leapfrog iterations for each HMC step.
  num_hmc_iters : int
    The number of HMC steps in this kernel.

  Returns
  -------
  samples_final : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    positions of the particles at the end of the kernel.
  float
    Empirical average acceptance rate of all HMC moves in this batch
    of particles.
  """
  step_keys = jax.random.split(key, num_hmc_iters)

  def short_hmc_step(loc_samples, loc_key):
    return hmc_step(loc_samples,
                    loc_key,
                    epsilon=epsilon,
                    log_density=log_density,
                    grad_log_density=grad_log_density,
                    num_leapfrog_iters=num_leapfrog_iters)

  samples_final, acceptance_rates = jax.lax.scan(short_hmc_step, samples_in,
                                                 step_keys)

  return samples_final, jnp.mean(acceptance_rates)

def hmc_wrapped(samples_in: jax.Array,
                key: jax.Array,
                epsilon: float,
                log_density_by_step: Callable[[float, jax.Array], jax.Array],
                beta: float,
                num_leapfrog_iters: int,
                num_hmc_iters: int
                ) -> Tuple[jax.Array, jax.Array]:
  """A wrapper for HMC that deals with all the interfacing with the codebase.

  Parameters
  ----------
  samples_in : jax.Array
    An array of shape (num_particles, particle_dim) containing 
    the positions of the particles.
  key : jax.Array
    A jax PRNG key.
  epsilon : float
    A Scalar representing the constant step size.
  log_density_by_step : Callable[[float, jax.Array], jax.Array]
    A function that takes in the current annealing temperature and 
    the array of particle positions, and returns the log densities 
    of each particle in an array under the current bridging distribution.
  beta : float
    The current annealing temperature.
  num_leapfrog_iters : int
    The number of leapfrog iterations for each HMC step.
  num_hmc_iters : int
    The number of HMC steps in this kernel.

  Returns
  -------
  samples_out : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    positions of the particles at the end of the kernel.
  acceptance : float
    Empirical average acceptance rate of all HMC moves in this batch
    of particles.
  """
  log_density = lambda x: log_density_by_step(beta, x)
  def unbatched_log_density(unbatched_tree_in):
    # Takes an unbatched tree and returns a single scalar value.
    batch_one_tree = jax.tree_util.tree_map(lambda x: x[None],
                                            unbatched_tree_in)
    return log_density(batch_one_tree)[0]

  grad_log_density = jax.vmap(jax.grad(unbatched_log_density))
  samples_out, acceptance = hmc(
      samples_in,
      key=key,
      epsilon=epsilon,
      log_density=log_density,
      grad_log_density=grad_log_density,
      num_leapfrog_iters=num_leapfrog_iters,
      num_hmc_iters=num_hmc_iters)
  return samples_out, acceptance

def hmc_scheduled_step_size(samples_in: jax.Array, 
                            key: jax.Array, 
                            log_density_by_step: Callable[[float, jax.Array], jax.Array], 
                            beta: float,
                            num_leapfrog_iters: int, 
                            num_hmc_iters: int, 
                            step_size_scheduler: InterpolatedStepSizeSchedule,
                            step: int) -> Tuple[jax.Array, jax.Array]:
  """A wrapper for HMC that uses interpolated step sizes.

  Parameters
  ----------
  samples_in : jax.Array
    An array of shape (num_particles, particle_dim) containing 
    the positions of the particles.
  key : jax.Array
    A jax PRNG key.
  log_density_by_step : Callable[[float, jax.Array], jax.Array]
    A function that takes in the current annealing temperature and 
    the array of particle positions, and returns the log densities 
    of each particle in an array under the current bridging distribution.
  beta : float
    The current annealing temperature.
  num_leapfrog_iters : int
    The number of leapfrog iterations for each HMC step.
  num_hmc_iters : int
    The number of HMC steps in this kernel.
  step_size_scheduler : InterpolatedStepSizeSchedule
    The step size scheduler.
  step : int
    the current iteration step index.

  Returns
  -------
  samples_out : jax.Array
    An array of shape (num_particles, particle_dim) containing the 
    positions of the particles at the end of the kernel.
  acceptance : float
    Empirical average acceptance rate of all HMC moves in this batch
    of particles.
  """
  epsilon = step_size_scheduler(step)
  samples_out, acceptance = hmc_wrapped(samples_in, key, epsilon, 
                                        log_density_by_step, beta, 
                                        num_leapfrog_iters, num_hmc_iters)
  return samples_out, acceptance