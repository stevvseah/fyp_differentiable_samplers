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
  config.threshold = 0.5
  config.num_temps = 11
  config.algo = 'aft'
  config.report_interval = 1
  
  # optional
  config.betas = None

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

  ###############
  # flow config #
  ###############
  flow_config = ConfigDict()

  flow_config.type = 'AffineInverseAutoregressiveFlow'
  flow_config.num_hidden_layers = 3
  flow_config.hidden_layer_dim = 30

  config.flow_config = flow_config

  ##############
  # aft config #
  ##############
  aft_config = ConfigDict()

  aft_config.num_train_iters = 200
  aft_config.train_num_particles = 6000
  aft_config.initial_learning_rate = 1e-3
  aft_config.boundaries_and_scales = None

  config.aft_config = aft_config

  ################
  # craft config #
  ################
  craft_config = ConfigDict()

  craft_config.num_train_iters = 200
  craft_config.initial_learning_rate = 5e-2
  craft_config.boundaries_and_scales = ({100: 1e-2},)

  config.craft_config = craft_config

  #~~~~~~~~~~~~~~~~~~~~#
  # end of config dict #
  #~~~~~~~~~~~~~~~~~~~~#

  return config