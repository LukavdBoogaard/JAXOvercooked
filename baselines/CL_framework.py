# Framework for continual learning experiments

import os
import sys
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax
import hydra
import distrax
import wandb
from omegaconf import OmegaConf

import jax_marl
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.environments.overcooked_environment import overcooked_layouts
from baselines.utils import Transition, batchify, unbatchify, pad_observation_space, sample_discrete_action, get_rollout_for_visualization, visualize_environments
from jax_marl.wrappers.baselines import LogWrapper

from baselines.IPPO_original import train



config_name = "ippo_continual" # REPLACE WITH YOUR CONFIG NAME



def make_train_fn(config):
    """
    Create the training function for the continual learning experiment.
    @param config: the configuration dictionary
    """

    def train_sequence(rng):
        '''
        Train on a sequence of tasks.
        @param rng: the random key for the experiment
        @return: the training output
        '''
        
        # Pad the environments to get a uniform shape
        padded_envs = pad_observation_space(config)
        envs = []
        for env_layout in padded_envs:
            # Create the environment object
            env = jax_marl.make(config["ENV_NAME"], layout=env_layout)
            env = LogWrapper(env, replace_info=False)
            envs.append(env)
        
        # add configuration items
        temp_env = envs[0]
        config["NUM_ACTORS"] = temp_env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

        # loop over environments
        rng, *env_rngs = jax.random.split(rng, len(envs)+1)
        
        def loop_over_envs(rng, train_state, envs):
            '''
            Loop over the environments in the sequence.
            @param rng: the random key for the experiment
            @param train_state: the current state of the training
            @param envs: the environments in the sequence
            @return: the updated training state
            '''
            for env, env_rng in zip(envs, env_rngs):
                runner_state, metric = 
                
                # get the training state
                train_state = train_state
            return train_state




    return train_sequence


@hydra.main(version_base=None, config_path="config", config_name=config_name)
def main(config):
    # set the device to GPU
    jax.config.update("jax_platform_name", "gpu")

    config = OmegaConf.to_container(config)

    # generate a sequence of tasks
    seq_length = config["SEQ_LENGTH"]
    strategy = config["STRATEGY"]
    config["ENV_KWARGS"], config["LAYOUT_NAME"] = generate_sequence(seq_length, strategy, layouts=None)

    # create the environments from the names in the sequence
    for layout_name in config["LAYOUT_NAME"]:
        layout_name = overcooked_layouts[layout_name]
    
    # Initialize WandB
    wandb.init(
        project='Continual_IPPO', 
        config=config,
        mode=config["WANDB_MODE"],
        name=f'{config["LAYOUT_NAME"]}_{config["SEQ_LENGTH"]}_{config["STRATEGY"]}'
    )

    # Create the training loop
    with jax.disable_jit():
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_jit = jax.jit(make_train_fn(config))
        output = jax.vmap(train_jit)(rngs)
    
    print("Finished training")

if __name__ == "__main__":
    main()