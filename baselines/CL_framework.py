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
import gc
from jax import clear_caches
from typing import Sequence, NamedTuple, Any
from functools import partial
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze

import jax_marl
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.environments.overcooked_environment import overcooked_layouts
from baselines.utils import Transition, batchify, unbatchify, pad_observation_space, sample_discrete_action, get_rollout_for_visualization, visualize_environments
from jax_marl.wrappers.baselines import LogWrapper
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from baselines.ippo_algorithm import Config, ippo_train
from baselines.algorithms import ActorCritic





def initialize_networks(config, env, rng):
    '''
    Initialize the appropriate networks based on the algorithm to be used
    @param config: the configuration dictionary
    @param env: the environment
    @return: the initialized networks
    '''

    print("In initializing networks")

    # get the algorithm
    algorithm = config.alg_name
     
    if algorithm == "ippo":
        network = ActorCritic(env.action_space().n, activation=config.activation)
        rng, network_rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape).flatten()
        network_params = network.init(network_rng, init_x)

        def linear_schedule(count):
            '''
            Linearly decays the learning rate depending on the number of minibatches and number of epochs
            returns the learning rate
            '''
            frac = 1.0 - ((count // (config.num_minibatches * config.update_epochs)) / config.num_updates)
            return config.lr * frac
        
        if config.anneal_lr: 
            # anneals the learning rate
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            # uses the default learning rate
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm), 
                optax.adam(config.lr, eps=1e-5)
            )

            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )

        return network, train_state, rng
    else:
        raise ValueError("Algorithm not recognized")
    

########################################################################################
########################################################################################
########################################################################################
########################################################################################

def make_train_fn(config):
    """
    Create the training function for the continual learning experiment.
    @param config: the configuration dictionary
    """

    @partial(jax.jit, static_argnums=(0,))
    def train_sequence(rng):
        '''
        Train on a sequence of tasks.
        @param rng: the random key for the experiment
        @return: the training output
        '''
        
        unfreeze(config)
        print(config)

        # Pad the environments to get a uniform shape
        padded_envs = pad_observation_space(config)
        envs = []
        for env_layout in padded_envs:
            # Create the environment object
            env = jax_marl.make(config["ENV_NAME"], layout=env_layout)
            env = LogWrapper(env, replace_info=False)
            envs.append(env)
        
        print("Created environments")
        
        # add configuration items
        temp_env = envs[0]
        config["NUM_ACTORS"] = temp_env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])


        freeze(config)
        print("Config is frozen")

        # REWARD SHAPING IN NEW VERSION
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.,
            end_value=0.,
            transition_steps=config["REWARD_SHAPING_HORIZON"]
        )

        # Initialize the network
        network, train_state, rng = initialize_networks(config, envs[0], rng)

        print("Initialized networks")

        # loop over environments
        rng, *env_rngs = jax.random.split(rng, len(envs)+1)
        
        def loop_over_envs(rng, network, train_state, envs, config):
            '''
            Loop over the environments in the sequence.
            @param rng: the random key for the experiment
            @param train_state: the current state of the training
            @param envs: the environments in the sequence
            @return: the updated training state
            '''
            print("In loop over envs")

            metrics = []
            for env, env_rng in zip(envs, env_rngs):
                print(f"Training on environment {env}")

                if config["ALG_NAME"] == "ippo":
                    runner_state, metric = ippo_train(network, train_state, env, env_rng, config)
                    metrics.append(metric)
                    train_state = runner_state[0]

                    print(f"Finished training on environment")
                else:
                    raise ValueError("Algorithm not recognized")

            return runner_state, metrics
        
        # apply the loop_over_envs function to the environments
        runner_state, metrics = loop_over_envs(rng, network, train_state, envs, config)

        return runner_state, metrics

    return train_sequence


config_name = "ippo_continual" # REPLACE WITH YOUR CONFIG NAME

@hydra.main(version_base=None, config_path="config", config_name=config_name)
def main(config):
    # set the device to GPU
    jax.config.update("jax_platform_name", "gpu")

    config = OmegaConf.to_container(config)

    # generate a sequence of tasks
    seq_length = config["SEQ_LENGTH"]
    strategy = config["STRATEGY"]
    config["ENV_KWARGS"], config["LAYOUT_NAME"] = generate_sequence(seq_length, strategy, layouts=None)


    for layout_config in config["ENV_KWARGS"]:
        # Extract the layout name
        layout_name = layout_config["layout"]

        # Set the layout in the config
        layout_config["layout"] = overcooked_layouts[layout_name]
    
    # Initialize WandB
    wandb.init(
        project='Continual_IPPO', 
        config=config,
        sync_tensorboard=True,
        mode=config["WANDB_MODE"],
        name=f'{config["LAYOUT_NAME"]}_{config["SEQ_LENGTH"]}_{config["STRATEGY"]}'
    )
    
    freeze(config)

    # Create the training loop
    with jax.disable_jit():
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_jit = jax.jit(make_train_fn(config))
        output = jax.vmap(train_jit)(rngs)
    
    print("Finished training")

if __name__ == "__main__":
    main()