"""Implementation of the HMC Markov kernel."""

import chex
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

class InterpolatedStepSize:
  config: ConfigDict
  total_num_time_steps: int