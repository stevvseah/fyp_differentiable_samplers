"""Trains a flow model by variational inference to learn two separate mappings based 
on the value of the annealing temperature."""

from absl import app, flags
import jax
from annealed_flow_transport.train import sample
from annealed_flow_transport import flows
from ml_collections.config_flags import config_flags
from matplotlib import pyplot as plt
import numpy as np
from typing import Sequence

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', './configs/special/two_gaussians.py', 
                                'Training configuration.')

def main(argv: Sequence[str]) -> None:
  config = FLAGS.config
  if len(argv) > 1:
    raise app.UsageError('Too many command line arguments.')
  log_evidence, main_output, misc = sample(config)

  # generate plot of particles
  flow = getattr(flows, 'TimeEmbedded' + config.flow_config.type)(config)
  key = jax.random.key(4199)
  key, key_ = jax.random.split(key)
  noise = jax.random.normal(key, (config.num_particles, config.particle_dim))
  gaussian1, _ = flow.apply(misc['params'], noise, 0.5, 0.)
  gaussian2, _ = flow.apply(misc['params'], noise, 1., 0.)

  plt.scatter([x[0] for x in noise], 
              [x[1] for x in noise], 
              marker='.', s=10, label='Original samples')
  plt.scatter([x[0] for x in gaussian1], 
              [x[1] for x in gaussian1], 
              marker='.', s=10, label='Generated samples\ntemperature = 1.0')
  plt.scatter([x[0] for x in gaussian2], 
              [x[1] for x in gaussian2], 
              marker='.', s=10, label='Generated samples\ntemperature = 0.5')
  plt.legend()

  if config.save_plot:
    plt.savefig(config.save_plot_path, bbox_inches='tight')
  plt.show()
  plt.close()

if __name__ == '__main__':
  app.run(main)