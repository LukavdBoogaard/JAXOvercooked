""" 
Based on PureJaxRL Implementation of PPO
"""

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

# Set the global config variable
config = None

class Transition(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray # whether the episode is done
    action: jnp.ndarray # the action taken
    value: jnp.ndarray # the value of the state
    reward: jnp.ndarray # the reward received
    log_prob: jnp.ndarray # the log probability of the action
    obs: jnp.ndarray # the observation
    # info: jnp.ndarray # additional information

############################
##### HELPER FUNCTIONS #####
############################

def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)

def get_rollout(config):
    '''
    Simulates the environment using the network
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the state sequence
    '''
    env = jax_marl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    rewards = []
    shaped_rewards = []
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # Get the action space for each agent (assuming it's uniform and doesn't depend on the agent_id)
        action_space_0 = env.action_space()  # Assuming the method needs to be called
        action_space_1 = env.action_space()  # Same as above since action_space is uniform

        # Sample actions for each agent
        action_0 = sample_discrete_action(key_a0, action_space_0).item()  # Ensure it's a Python scalar
        action_1 = sample_discrete_action(key_a1, action_space_1).item()

        actions = {
            "agent_0": action_0,
            "agent_1": action_1
        }

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]
        rewards.append(reward["agent_0"])
        shaped_rewards.append(info["shaped_reward"]["agent_0"])

        state_seq.append(state)

    return state_seq


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked") 
def main(cfg):
    # check available devices
    print(jax.devices())
    
    # set the config to global 
    global config

    # convert the config to a dictionary
    config = OmegaConf.to_container(cfg)

    # set the layout of the environment
    layout = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout]

    filename = f'{config["ENV_NAME"]}_{layout}'
    state_seq = get_rollout(config)
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

if __name__ == "__main__":
    print("Running main...")
    main()

