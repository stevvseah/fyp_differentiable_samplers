"""Trains a flow model by variational inference to learn two separate mappings based 
on the value of the annealing temperature."""

from ml_collections import ConfigDict

def get_config():
  config = ConfigDict()

  ###############
  # main config #
  ###############
  config.seed = 1
  config.save_plot = False
  config.save_plot_path = 'images/time_embedding_example'
  config.num_particles = 2000
  config.particle_dim = 2
  config.threshold = 0.3
  config.num_temps = 11
  config.algo = 'vi'
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

  ############################################
  # final density config (these are ignored) #
  ############################################
  final_density_config = ConfigDict()

  final_density_config.density = 'NormalDistribution'
  final_density_config.loc = 5.
  final_density_config.scale = 1.

  config.final_density_config = final_density_config

  ################################
  # kernel config (also ignored) #
  ################################
  kernel_config = ConfigDict()

  kernel_config.num_leapfrog_iters = 10
  kernel_config.num_hmc_iters = 1
  kernel_config.step_size = 0.2

  config.kernel_config = kernel_config

  ###############
  # flow config #
  ###############
  flow_config = ConfigDict()

  flow_config.type = 'DiagonalAffine'
  flow_config.time_dim = 10
  flow_config.num_coupling_layers = 3
  flow_config.num_hidden_layers_per_coupling = 2
  flow_config.hidden_layer_dim = 5

  config.flow_config = flow_config

  #############
  # vi config #
  #############
  vi_config = ConfigDict()

  vi_config.num_train_iters = 5
  vi_config.initial_learning_rate = 1e-1
  vi_config.boundaries_and_scales = None
  vi_config.beta_list = [[0., 0.5], [0., 1.]] * 100
  vi_config.embed_time = True

  vi_config.special_target_density = 'TwoGaussians'

  config.vi_config = vi_config

  #~~~~~~~~~~~~~~~~~~~~#
  # end of config dict #
  #~~~~~~~~~~~~~~~~~~~~#

  return config