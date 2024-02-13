"""Runs a sampler over Neal's funnel distribution."""

from ml_collections import ConfigDict

def get_config():
  config = ConfigDict()

  ###############
  # main config #
  ###############
  config.seed = 1
  config.num_particles = 2000
  config.particle_dim = 10
  config.threshold = 0.3
  config.num_temps = 2
  config.algo = 'smc'
  config.report_interval = 1
  
  # optional
  config.betas = None

  # for running on experiment.py
  config.repetitions = 200
  config.save_results_path = 'results/funnel_smc1.csv'

  ##########################
  # initial density config #
  ##########################
  initial_density_config = ConfigDict()

  initial_density_config.loc = 0.
  initial_density_config.scale = 1.

  config.initial_density_config = initial_density_config

  ########################
  # final density config #
  ########################
  final_density_config = ConfigDict()

  final_density_config.density = 'NealsFunnel'


  config.final_density_config = final_density_config

  #################
  # kernel config #
  #################
  kernel_config = ConfigDict()

  kernel_config.num_leapfrog_iters = 10
  kernel_config.num_hmc_iters = 1
  kernel_config.step_size = 0.2

  # optional
  kernel_config.interp_step_times = [0., 0.25, 0.5, 0.75, 1.]
  kernel_config.interp_step_sizes = [0.9, 0.7, 0.6, 0.5, 0.4]

  config.kernel_config = kernel_config

  ##############
  # smc config #
  ##############
  smc_config = ConfigDict()
  
  config.smc_config = smc_config

  #~~~~~~~~~~~~~~~~~~~~#
  # end of config dict #
  #~~~~~~~~~~~~~~~~~~~~#

  return config