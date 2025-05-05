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

class Transition_CNN(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray # whether the episode is done
    action: jnp.ndarray # the action taken
    value: jnp.ndarray # the value of the state
    reward: jnp.ndarray # the reward received
    log_prob: jnp.ndarray # the log probability of the action
    obs: jnp.ndarray # the observation
    info: jnp.ndarray # additional information

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

def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)


def make_task_onehot(task_idx: int, num_tasks: int) -> jnp.ndarray:
    """
    Returns a one-hot vector of length `num_tasks` with a 1 at `task_idx`.
    """
    return jnp.eye(num_tasks, dtype=jnp.float32)[task_idx]

def copy_params(params):
    return jax.tree_util.tree_map(lambda x: x.copy(), params)

def compute_fwt(matrix):
    """
    Computes the forward transfer for all tasks in a sequence
    param matrix: a 2D array of shape (num_tasks + 1, num_tasks) where each entry is the performance of the model on the task
    """
    # Assert that the matrix has the correct shape
    assert matrix.shape[0] == matrix.shape[1] + 1, "Matrix must have shape (num_tasks + 1, num_tasks)"

    num_tasks = matrix.shape[1]

    # Create a 2D array to store the forward transfer values
    fwt = jnp.full((num_tasks,), jnp.nan)

    for i in range(num_tasks):
        # the first task has no forward transfer
        if i == 0:
            continue
        # Compute the forward transfer for task i
        fwt = fwt.at[i].set(matrix[i, i] - matrix[0, i]) 

    avg_fwt = jnp.nanmean(fwt)
    return fwt, avg_fwt

def compute_bwt(matrix):
    """
    Computes the backward transfer for all tasks in a sequence
    param matrix: a 2D array of shape (num_tasks + 1, num_tasks) where each entry is the performance of the model on the task
    """
    # Assert that the matrix has the correct shape
    assert matrix.shape[0] == matrix.shape[1] + 1, "Matrix must have shape (num_tasks + 1, num_tasks)"

    num_tasks = matrix.shape[1]
    bwt_series = []
    # Create a 2D array to store the backward transfer values
    bwt_avg = jnp.full((num_tasks,), jnp.nan)

    for i in range(num_tasks-1):
        difference = matrix[i+2, i+1] - matrix[i+1,i+1]
        bwt_series.append(difference)
        bwt_avg = bwt_avg.at[i].set(jnp.nanmean(difference))

    return bwt_series, bwt_avg

