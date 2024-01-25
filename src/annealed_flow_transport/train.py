"""Management of interactions with config scripts."""

import jax
import optax
from ml_collections import ConfigDict
from typing import Callable, Tuple, Any
import flax.linen as nn
from . import densities, flows
from .samplers import NormalSampler
from .utils.aft_types import InitialDensitySampler, LogDensityByTemp, LogDensity
from .utils.aft_types import InterpolatedStepSizeSchedule
from .densities import NormalDistribution, NealsFunnel, ChallengingTwoDimensionalMixture
from .utils.hmc import HMCKernel
from . import smc, aft, craft

def value_or_none(value: str, config: ConfigDict) -> Any:
  """Looks for a desired attribute in the input ConfigDict 
  and outputs its stored value, or None if no attribute exists.
  
  Parameters
  ----------
  value : str
    The desired attribute.
  config : ConfigDict
    The ConfigDict to search for value for.

  Returns
  -------
  Any
    Either the stored value of the desired attribute, or 
    None if the attribute does not exist in config.
  """
  if value in config:
    return config[value]
  else:
    return None

def get_optimizer(initial_learning_rate: float,
                  boundaries_and_scales: Tuple | None
                  ) -> optax.GradientTransformation:
  """Get an optimizer possibly with learning rate schedule.
  
  Parameters
  ----------
  initial_learning_rate : float
    The desired initial learning rate of an ADAM optimizer.
  boundaries_and_scales : Tuple | None
    A tuple containing a dictionary with key-value pairs 
    of the form {`step`: `learning_rate`}, where `step` 
    is an integer value of the iteration time step, and 
    `learning_rate` is the desired new learning rate to 
    apply at that time step. If None, the output 
    optimizer simply has constant learning rate.

  Returns
  -------
  opt : optax.GradientTransformation
    The resulting optimizer as specified by the inputs.
  """
  if boundaries_and_scales is None:
    return optax.adam(initial_learning_rate)
  else:
    schedule_fn = optax.piecewise_constant_schedule(
        initial_learning_rate,
        boundaries_and_scales[0])
    opt = optax.chain(optax.scale_by_adam(),
                      optax.scale_by_schedule(schedule_fn), optax.scale(-1.))
    return opt

def sample(config: ConfigDict):
  """Accesses the input config dict to configure a sampling 
  algorithm and run it.

  Parameters
  ----------
  config : ConfigDict

  Returns
  -------
  log_evidence : float
    The log evidence estimate produced from the sampler.
  main_output : dict
    A dictionary containing the produced samples, their 
    log weights, and the acceptance rate history of the 
    HMC kernel.
  misc : dict
    A dictionary containing miscellaneous output of the 
    sampler.
  """
  num_particles = config.num_particles
  particle_dim = config.particle_dim
  threshold = config.threshold
  num_temps = config.num_temps
  betas = value_or_none('temperature_schedule', config)
  report_interval = config.report_interval

  # initial distribution config
  loc = config.initial_density_config.loc
  scale = config.initial_density_config.scale
  initial_log_density = NormalDistribution(config.initial_density_config)
  sampler = NormalSampler(num_particles, particle_dim, loc, scale)
  final_log_density = getattr(densities, 
                              config.final_density_config.density)(
                                config.final_density_config)
  log_density = LogDensityByTemp(initial_log_density, final_log_density)

  # kernel config
  num_leapfrog_iters = config.kernel_config.num_leapfrog_iters
  num_hmc_iters = config.kernel_config.num_hmc_iters
  if hasattr(config, 'interp_step_times') and hasattr(config, 'interp_step_sizes'):
    epsilon = InterpolatedStepSizeSchedule(config.kernel_config.interp_step_times,  
                                           config.kernel_config.interp_step_sizes, 
                                           num_temps)
  else: 
    epsilon = config.kernel_config.step_size
  kernel = HMCKernel(log_density, epsilon, num_leapfrog_iters, num_hmc_iters)

  key = jax.random.key(config.seed)
  key, key_ = jax.random.split(key)

  if config.algo == 'smc':
    samples, log_weights, log_evidence, acpt_rate = smc.apply(key_, log_density, 
                                                              sampler, kernel, 
                                                              threshold, num_temps, 
                                                              betas, report_interval)
    misc = {}
  
  elif config.algo == 'aft':

    aft_num_train_iters = config.aft_config.num_train_iters
    train_batch_size = config.aft_config.train_num_particles
    train_sampler = NormalSampler(train_batch_size, particle_dim)

    initial_learning_rate = config.aft_config.initial_learning_rate
    boundaries_and_scales = value_or_none('boundaries_and_scales', config.aft_config)
    opt = get_optimizer(initial_learning_rate, boundaries_and_scales)

    embed_time = config.aft_config.embed_time
    refresh_opt_state = config.aft_config.refresh_opt_state

    if embed_time:
      flow = getattr(flows, 'TimeEmbedded' + config.flow_config.type)(config)
      params = flow.init(key, sampler(key), 0.1, 0.)
    else:
      flow = getattr(flows, config.flow_config.type)(config)
      params = flow.init(key, sampler(key))

    flow_apply = flow.apply

    samples, log_weights, log_evidence, acpt_rate, \
    val_loss_history, train_loss_history = aft.apply(key_, log_density, sampler, 
                                                     train_sampler, kernel, 
                                                     flow_apply, params, 
                                                     opt, threshold, 
                                                     aft_num_train_iters, 
                                                     num_temps, betas, 
                                                     report_interval, 
                                                     embed_time, 
                                                     refresh_opt_state)
    misc = {'val_loss': val_loss_history, 'train_loss': train_loss_history}

  elif config.algo == 'craft':

    craft_num_train_iters = config.craft_config.num_train_iters

    initial_learning_rate = config.craft_config.initial_learning_rate
    boundaries_and_scales = value_or_none('boundaries_and_scales', config.craft_config)
    opt = get_optimizer(initial_learning_rate, boundaries_and_scales)

    embed_time = config.craft_config.embed_time

    if embed_time:
      flow = getattr(flows, 'TimeEmbedded' + config.flow_config.type)(config)
      params = flow.init(key, sampler(key), 0.1, 0.)
    else:
      flow = getattr(flows, config.flow_config.type)(config)
      params = flow.init(key, sampler(key))

    flow_apply = flow.apply

    samples, log_weights, acpt_rate, log_evidence, \
    train_loss_history, log_evidence_history = craft.apply(key_, sampler, flow_apply, 
                                                           params, opt, log_density, 
                                                           kernel, num_temps, threshold, 
                                                           craft_num_train_iters, betas, 
                                                           report_interval, embed_time)
    misc = {'train_loss': train_loss_history, 'evidence_hist': log_evidence_history}
  else:
    raise NotImplementedError
  
  main_output = {'samples': samples, 'log_weights': log_weights, 'acpt_rate': acpt_rate}

  return log_evidence, main_output, misc