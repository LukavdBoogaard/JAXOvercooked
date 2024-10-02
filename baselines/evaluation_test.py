import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jax_marl
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import wandb




def evaluation():
    return None

def training_loop():

    def step(carry, _):
        grads = carry * 2 + 5 / 2.4

        return grads, None


    carry = 0
    carry, result = jax.lax.scan(
        f=step,
        init=carry, 
        xs=None,
        length=100)
    
    

def main():
    prng = jax.random.PRNGKey(0)
    prng, subkey = jax.random.split(prng)

    # jit the training loop
    training_loop = jax.jit(training_loop)
