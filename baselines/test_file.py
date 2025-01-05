# test file to test various things 


import os
import wandb
from dotenv import load_dotenv
import jax
from dataclasses import dataclass
from functools import partial
from flax.core.frozen_dict import FrozenDict


layouts = [FrozenDict()]