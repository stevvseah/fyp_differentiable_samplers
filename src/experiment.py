"""Runs a sampling experiment (as defined in the input config file) multiple times and 
saves the output log evidence in a csv file at a specified location."""

from absl import app, flags
from annealed_flow_transport import train
from ml_collections.config_flags import config_flags
from typing import Sequence
import csv

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', './configs/funnel.py',
                                'Training configuration.')

def main(argv: Sequence[str]) -> None:
  config = FLAGS.config
  if len(argv) > 1:
    raise app.UsageError('Too many command line arguments.')
  for _ in range(config.repetitions):
    results = train.sample(config)
    with open(config.save_results_path, 'a', newline='') as f:
      writer = csv.writer(f)
      if 'temperatures' not in results[-1]:
        writer.writerow(results[:2])
      else:
        writer.writerow([results[0], results[1], 
                         len(results[-1]['temperatures'])])
    config.seed += 1

if __name__ == '__main__':
  app.run(main)