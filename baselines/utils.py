import jax
import jax.numpy as jnp
from typing import NamedTuple, Any
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import Module
import numpy as np
from functools import partial

import jax_marl
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer



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

def batchify(x: dict, agent_list, num_actors):
    '''
    converts the observations of a batch of agents into an array of size (num_actors, -1) that can be used by the network
    @param x: dictionary of observations
    @param agent_list: list of agents
    @param num_actors: number of actors
    returns the batchified observations
    '''
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    '''
    converts the array of size (num_actors, -1) into a dictionary of observations for all agents
    @param x: array of observations
    @param agent_list: list of agents
    @param num_envs: number of environments
    @param num_actors: number of actors
    returns the unbatchified observations
    '''
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def pad_observation_space(config):
    '''
    Pads the observation space of the environment to be compatible with the network
    @param envs: the environment
    returns the padded observation space
    '''

    print('in pad_observation_space')

    envs = []
    for env_args in config["ENV_KWARGS"]:
            # Create the environment
            env = jax_marl.make(config["ENV_NAME"], **env_args)
            envs.append(env)

    # find the environment with the largest observation space
    max_width, max_height = 0, 0
    for env in envs:
        max_width = max(max_width, env.layout["width"])
        max_height = max(max_height, env.layout["height"])
    
    # pad the observation space of all environments to be the same size by adding extra walls to the outside
    padded_envs = []
    for env in envs:
        # unfreeze the environment so that we can apply padding
        env = unfreeze(env.layout)  

        # calculate the padding needed
        width_diff = max_width - env["width"]
        height_diff = max_height - env["height"]

        # determine the padding needed on each side
        left = width_diff // 2
        right = width_diff - left
        top = height_diff // 2
        bottom = height_diff - top

        width = env["width"]

        # Adjust the indices of the observation space to match the padded observation space
        def adjust_indices(indices):
            '''
            adjusts the indices of the observation space
            @param indices: the indices to adjust
            returns the adjusted indices
            '''
            adjusted_indices = []

            for idx in indices:
                # Compute the row and column of the index
                row = idx // width
                col = idx % width
                
                # Shift the row and column by the padding
                new_row = row + top
                new_col = col + left
                
                # Compute the new index
                new_idx = new_row * (width + left + right) + new_col
                adjusted_indices.append(new_idx)
            
            return jnp.array(adjusted_indices)
        
        # adjust the indices of the observation space to account for the new walls
        env["wall_idx"] = adjust_indices(env["wall_idx"])
        env["agent_idx"] = adjust_indices(env["agent_idx"])
        env["goal_idx"] = adjust_indices(env["goal_idx"])
        env["plate_pile_idx"] = adjust_indices(env["plate_pile_idx"])
        env["onion_pile_idx"] = adjust_indices(env["onion_pile_idx"])
        env["pot_idx"] = adjust_indices(env["pot_idx"])

        # pad the observation space with walls
        padded_wall_idx = list(env["wall_idx"])  # Existing walls
        
        # Top and bottom padding
        for y in range(top):
            for x in range(max_width):
                padded_wall_idx.append(y * max_width + x)  # Top row walls

        for y in range(max_height - bottom, max_height):
            for x in range(max_width):
                padded_wall_idx.append(y * max_width + x)  # Bottom row walls

        # Left and right padding
        for y in range(top, max_height - bottom):
            for x in range(left):
                padded_wall_idx.append(y * max_width + x)  # Left column walls

            for x in range(max_width - right, max_width):
                padded_wall_idx.append(y * max_width + x)  # Right column walls

        env["wall_idx"] = jnp.array(padded_wall_idx)

        # set the height and width of the environment to the new padded height and width
        env["height"] = max_height
        env["width"] = max_width

        padded_envs.append(freeze(env)) # Freeze the environment to prevent further modifications
        
    return padded_envs

def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)

def get_rollout_for_visualization(config):
    '''
    Simulates the environment using the network
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the state sequence
    '''

    # Add the padding
    envs = pad_observation_space(config)

    state_sequences = []
    for env_layout in envs:
        env = jax_marl.make(config["ENV_NAME"], layout=env_layout)

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
        state_sequences.append(state_seq)

    return state_sequences
    
def visualize_environments(config):
    '''
    Visualizes the environments using the OvercookedVisualizer
    @param config: the configuration of the training
    returns None
    '''
    state_sequences = get_rollout_for_visualization(config)
    visualizer = OvercookedVisualizer()
    visualizer.animate(state_seq=state_sequences[0], agent_view_size=5, filename="initial_state_env1.gif")
    visualizer.animate(state_seq=state_sequences[1], agent_view_size=5, filename="initial_state_env2.gif")

    return None
