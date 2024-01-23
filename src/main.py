"""Main entry point for running the samplers in this repository."""

from absl import app, flags
from annealed_flow_transport import train
from ml_collections.config_flags import config_flags
from typing import Sequence

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', './configs/simple_normal.py',
                                'Training configuration.')

def main(argv: Sequence[str]) -> None:
  config = FLAGS.config
  if len(argv) > 1:
    raise app.UsageError('Too many command line arguments.')
  results = train.sample(config)

if __name__ == '__main__':
  app.run(main)