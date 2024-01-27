"""Trains a flow model (with its reverse implemented) to generate from the 
Sci-kit Learn two moons dataset.

This config file is not meant to be used with main.py!
"""

from ml_collections import ConfigDict

def get_config():  
  config = ConfigDict()

  ###############
  # main config #
  ###############
  config.seed = 10
  config.data_file_path = './data/two_moons.csv'
  config.save_plot = False
  config.save_plot_path = './images/two_moons.png'
  config.num_particles = 2000
  config.particle_dim = 2
  config.report_interval = 1

  # optimizer_config #
  config.initial_learning_rate = 5e-3
  config.boundaries_and_scales = None

  # training config #
  config.num_train_iters = 2000
  config.train_batch_size = 512
  config.report_interval = 100

  ##########################
  # initial density config #
  ##########################
  initial_density_config = ConfigDict()

  initial_density_config.loc = 0.
  initial_density_config.scale = 1.

  config.initial_density_config = initial_density_config

  ###############
  # flow config #
  ###############
  flow_config = ConfigDict()

  flow_config.type = 'InvertibleRealNVP'
  flow_config.num_coupling_layers = 5
  flow_config.num_hidden_layers_per_coupling = 5
  flow_config.hidden_layer_dim = 32

  config.flow_config = flow_config

  #~~~~~~~~~~~~~~~~~~~~#
  # end of config dict #
  #~~~~~~~~~~~~~~~~~~~~#

  return config