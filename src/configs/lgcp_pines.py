"""Runs a sampler over the log-Gaussian-Cox process target distribution."""

from ml_collections import ConfigDict

def get_config():
  config = ConfigDict()

  ###############
  # main config #
  ###############
  config.seed = 1
  config.num_particles = 200
  config.particle_dim = 1024
  config.threshold = 0.3
  config.num_temps = 11
  config.algo = 'craft'
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

  final_density_config.density = 'LogGaussianCoxPines'
  final_density_config.particle_dim = config.particle_dim
  final_density_config.file_path = './data/finpines.csv'
  final_density_config.use_whitened = False


  config.final_density_config = final_density_config

  #################
  # kernel config #
  #################
  kernel_config = ConfigDict()

  kernel_config.num_leapfrog_iters = 10
  kernel_config.num_hmc_iters = 1
  kernel_config.step_size = 0.2

  # optional
  # kernel_config.interp_step_times = [0., 0.25, 0.5, 1.]
  # kernel_config.interp_step_sizes = [0.3, 0.3, 0.2, 0.2]

  config.kernel_config = kernel_config

  ##############
  # smc config #
  ##############
  smc_config = ConfigDict()

  smc_config.adaptive = True
  smc_config.num_adaptive_search_iters = 100
  smc_config.adaptive_threshold = 0.5
  
  config.smc_config = smc_config

  ###############
  # flow config #
  ###############
  flow_config = ConfigDict()

  flow_config.type = 'DiagonalAffine'
  flow_config.num_coupling_layers = 2
  flow_config.num_hidden_layers_per_coupling = 1
  flow_config.hidden_layer_dim = 200
  flow_config.time_dim = 8

  config.flow_config = flow_config

  ##############
  # aft config #
  ##############
  aft_config = ConfigDict()

  aft_config.num_train_iters = 100
  aft_config.train_num_particles = 200
  aft_config.initial_learning_rate = 1e-3
  aft_config.boundaries_and_scales = None
  aft_config.embed_time = True
  aft_config.refresh_opt_state = not aft_config.embed_time

  # adaptive config
  aft_config.adaptive = True
  aft_config.adaptive_with_flow = True
  aft_config.num_adaptive_search_iters = 50
  aft_config.adaptive_threshold = 0.3

  config.aft_config = aft_config

  ################
  # craft config #
  ################
  craft_config = ConfigDict()

  craft_config.num_train_iters = 60
  craft_config.initial_learning_rate = 1e-3
  craft_config.boundaries_and_scales = ({80: 5e-4},)
  craft_config.embed_time = True

  # adaptive config
  craft_config.adaptive = True
  craft_config.max_adaptive_num_temps = 11
  craft_config.num_adaptive_search_iters = 50
  craft_config.adaptive_threshold = 0.3

  config.craft_config = craft_config

  #############
  # vi config #
  #############
  vi_config = ConfigDict()

  vi_config.num_train_iters = 2000
  vi_config.initial_learning_rate = 1e-3
  vi_config.boundaries_and_scales = None
  vi_config.beta_list = [[0., 1.]]
  vi_config.embed_time = False

  config.vi_config = vi_config

  #~~~~~~~~~~~~~~~~~~~~#
  # end of config dict #
  #~~~~~~~~~~~~~~~~~~~~#

  return config