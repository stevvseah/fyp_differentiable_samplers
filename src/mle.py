"""Trains a flow model by MLE. Note that the reverse of the flow must be implemented first."""

from absl import app, flags
from annealed_flow_transport.train_mle import train
from ml_collections.config_flags import config_flags
from matplotlib import pyplot as plt
import numpy as np
from typing import Sequence

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', './configs/two_moons.py', 
                                'Training configuration.')

def main(argv: Sequence[str]) -> None:
  config = FLAGS.config
  if len(argv) > 1:
    raise app.UsageError('Too many command line arguments.')
  generated_samples, _, _, _ = train(config)

  with open('data/two_moons.csv', 'r') as f:
    original_samples = np.genfromtxt(f, delimiter=',')

  plt.scatter([x[0] for x in original_samples], 
              [x[1] for x in original_samples], 
              marker='.', s=10, label='Original samples')
  plt.scatter([x[0] for x in generated_samples], 
              [x[1] for x in generated_samples], 
              marker='.', s=10, label='Generated samples')
  plt.xlim([-2, 2])
  plt.ylim([-2, 2])
  plt.legend()

  if config.save_plot:
    plt.savefig(config.save_plot_path, bbox_inches='tight')
  plt.show()
  plt.close()

if __name__ == '__main__':
  app.run(main)