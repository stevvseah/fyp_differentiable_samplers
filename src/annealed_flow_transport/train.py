"""Management of interactions with config scripts."""

import jax
import optax
from ml_collections import ConfigDict
from typing import Tuple, Any
from . import densities, flows, special_target_densities
from .samplers import NormalSampler
from .utils.aft_types import LogDensityByTemp
from .utils.aft_types import InterpolatedStepSizeSchedule
from .densities import NormalDistribution
from .utils.hmc import HMCKernel
from . import smc, aft, craft, vi, adaptive_craft

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

def sample(config: ConfigDict) -> Tuple[float, float, dict, dict]:
  """Accesses the input config dict to configure a sampling 
  algorithm and run it.

  Parameters
  ----------
  config : ConfigDict
    The config dict specifying the configurations of the 
    training and algorithm.

  Returns
  -------
  log_evidence : float
    The log evidence estimate produced from the sampler.
  sampling_time : float
    The time taken to perform sampling.
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
  report_interval = config.report_interval

  if config.algo != 'vi':
    threshold = config.threshold
    num_temps = value_or_none('num_temps', config)
    betas = value_or_none('temperature_schedule', config)

  # initial distribution config
  loc = config.initial_density_config.loc
  scale = config.initial_density_config.scale
  initial_log_density = NormalDistribution(config.initial_density_config)
  sampler = NormalSampler(num_particles, particle_dim, loc, scale)
  if hasattr(config, 'special_target_density'):
    log_density = getattr(special_target_densities, 
                          config.special_target_density)(initial_log_density)
  else:
    final_log_density = getattr(densities, 
                                config.final_density_config.density)(
                                  config.final_density_config)
    log_density = LogDensityByTemp(initial_log_density, final_log_density)

  # kernel config
  if config.algo != 'vi':
    num_leapfrog_iters = config.kernel_config.num_leapfrog_iters
    num_hmc_iters = config.kernel_config.num_hmc_iters
    if hasattr(config.kernel_config, 'interp_step_times') \
      and hasattr(config.kernel_config, 'interp_step_sizes'):
      epsilon = InterpolatedStepSizeSchedule(config.kernel_config.interp_step_times,  
                                            config.kernel_config.interp_step_sizes, 
                                            num_temps)
    else: 
      epsilon = config.kernel_config.step_size
    kernel = HMCKernel(log_density, epsilon, num_leapfrog_iters, num_hmc_iters)

  key = jax.random.key(config.seed)
  key, key_ = jax.random.split(key)

  if config.algo == 'smc':
    if value_or_none('adaptive', config.smc_config):
      num_search_iters = config.smc_config.num_adaptive_search_iters
      adaptive_threshold = config.smc_config.adaptive_threshold
      samples, log_weights, log_evidence, \
      acpt_rate, beta_history, sampling_time = smc.apply_adaptive(key_, log_density, sampler, kernel, 
                                                                  threshold, report_interval, 
                                                                  num_search_iters, adaptive_threshold)
      misc = {'temperatures': beta_history}
    else:
      samples, log_weights, log_evidence, acpt_rate, sampling_time = smc.apply(key_, log_density, 
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

    if value_or_none('adaptive', config.aft_config):
      num_search_iters = config.aft_config.num_adaptive_search_iters
      adaptive_threshold = config.aft_config.adaptive_threshold
      adaptive_with_flow = config.aft_config.adaptive_with_flow
      samples, log_weights, log_evidence, acpt_rate, val_loss_history, \
      train_loss_history, beta_history, sampling_time = aft.apply_adaptive(key_, log_density, sampler, 
                                                                           train_sampler, kernel, 
                                                                           flow_apply, params, opt, 
                                                                           threshold, aft_num_train_iters, 
                                                                           report_interval, embed_time, 
                                                                           refresh_opt_state, 
                                                                           adaptive_with_flow, 
                                                                           num_search_iters, 
                                                                           adaptive_threshold)
      misc = {'val_loss': val_loss_history, 'train_loss': train_loss_history, 
              'temperatures': beta_history}
    else:
      samples, log_weights, log_evidence, acpt_rate, \
      val_loss_history, train_loss_history, sampling_time = aft.apply(key_, log_density, sampler, 
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

    if value_or_none('adaptive', config.craft_config):
      num_search_iters = config.craft_config.num_adaptive_search_iters
      adaptive_threshold = config.craft_config.adaptive_threshold
      max_adaptive_num_temps = config.craft_config.max_adaptive_num_temps
      samples, log_weights, acpt_rate, log_evidence, \
      train_loss_history, log_evidence_history, \
      beta_history, num_temps, sampling_time = adaptive_craft.apply(key_, sampler, flow_apply, 
                                                                    params, opt, kernel, 
                                                                    log_density, threshold, 
                                                                    num_search_iters, 
                                                                    adaptive_threshold, 
                                                                    max_adaptive_num_temps, 
                                                                    craft_num_train_iters, 
                                                                    report_interval)
      misc = {'train_loss': train_loss_history, 'evidence_hist': log_evidence_history, 
              'temperatures': beta_history, 'num_temps': num_temps}
    else:
      samples, log_weights, acpt_rate, log_evidence, \
      train_loss_history, log_evidence_history, sampling_time = craft.apply(key_, sampler, flow_apply, 
                                                                            params, opt, log_density, 
                                                                            kernel, num_temps, threshold, 
                                                                            craft_num_train_iters, betas, 
                                                                            report_interval, embed_time)
      misc = {'train_loss': train_loss_history, 'evidence_hist': log_evidence_history}

  elif config.algo == 'vi':
    
    vi_num_train_iters = config.vi_config.num_train_iters

    initial_learning_rate = config.vi_config.initial_learning_rate
    boundaries_and_scales = value_or_none('boundaries_and_scales', config.vi_config)
    opt = get_optimizer(initial_learning_rate, boundaries_and_scales)

    embed_time = config.vi_config.embed_time

    if embed_time:
      flow = getattr(flows, 'TimeEmbedded' + config.flow_config.type)(config)
      params = flow.init(key, sampler(key), 0.1, 0.)
      opt_state = opt.init(params)
    else:
      flow = getattr(flows, config.flow_config.type)(config)
      params = flow.init(key, sampler(key))
      opt_state = opt.init(params)

    flow_apply = flow.apply

    for beta_prev, beta in config.vi_config.beta_list:
      samples, log_weights, log_evidence, vfe_history, \
      log_evidence_history, params, opt_state, sampling_time = vi.apply(key_, params, beta, beta_prev, 
                                                                        opt_state, opt, sampler, 
                                                                        log_density, flow_apply, 
                                                                        embed_time, vi_num_train_iters, 
                                                                        report_interval)
    acpt_rate = None
    misc = {'vfe_history': vfe_history, 'evidence_hist': log_evidence_history, 
              'params': params, 'opt_state': opt_state}

  else:
    raise NotImplementedError
  
  main_output = {'samples': samples, 'log_weights': log_weights, 'acpt_rate': acpt_rate}

  return log_evidence, sampling_time, main_output, misc