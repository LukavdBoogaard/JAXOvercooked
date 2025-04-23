# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from datetime import datetime

import copy
from datetime import datetime
import pickle
import flax
import jax
import jax.experimental
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.linen.initializers import constant, orthogonal
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from typing import Sequence, NamedTuple, Any, Optional, List
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper

from jax_marl.registration import make
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
from jax_marl.environments.overcooked_environment.layouts import counter_circuit_grid
from dotenv import load_dotenv
import hydra
import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import wandb
from functools import partial
from dataclasses import dataclass, field
import tyro
from tensorboardX import SummaryWriter
from pathlib import Path

# Enable compile logging
# jax.log_compiles(True)

# class ActorCritic(nn.Module):
#     '''
#     Class to define the actor-critic networks used in IPPO. Each agent has its own actor-critic network
#     '''
#     action_dim: Sequence[int]
#     activation: str = "tanh"

#     @nn.compact
#     def __call__(self, x):
#         if self.activation == "relu":
#             activation = nn.relu
#         else:
#             activation = nn.tanh

#         # ACTOR  
#         actor_mean = nn.Dense(
#             128, # number of neurons
#             kernel_init=orthogonal(np.sqrt(2)), 
#             bias_init=constant(0.0) # sets the bias initialization to a constant value of 0
#         )(x) # applies a dense layer to the input x

#         actor_mean = activation(actor_mean) # applies the activation function to the output of the dense layer

#         actor_mean = nn.Dense(
#             128, 
#             kernel_init=orthogonal(np.sqrt(2)), 
#             bias_init=constant(0.0)
#         )(actor_mean)

#         actor_mean = activation(actor_mean)

#         actor_mean = nn.Dense(
#             self.action_dim, 
#             kernel_init=orthogonal(0.01), 
#             bias_init=constant(0.0)
#         )(actor_mean)

#         pi = distrax.Categorical(logits=actor_mean) # creates a categorical distribution over all actions (the logits are the output of the actor network)

#         # CRITIC
#         critic = nn.Dense(
#             128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(x)

#         critic = activation(critic)

#         critic = nn.Dense(
#             128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(critic)

#         critic = activation(critic)

#         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
#             critic
#         )
        
#         # returns the policy (actor) and state-value (critic) networks
#         value = jnp.squeeze(critic, axis=-1)
#         return pi, value #squeezed to remove any unnecessary dimensions

class Actor(nn.Module):
    """
    Actor network for MAPPO.
    
    This network takes observations as input and outputs a 
    categorical distribution over actions.
    """
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        # Choose the activation function based on input parameter.
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # First hidden layer
        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = act_fn(x)

        # Second hidden layer
        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = act_fn(x)

        # Output layer to produce logits for the action distribution
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(x)

        # Create a categorical distribution using the logits
        pi = distrax.Categorical(logits=logits)
        return pi

class Critic(nn.Module):
    '''
    Critic network that estimates the value function
    '''
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        # Choose activation function
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        # First hidden layer
        critic = nn.Dense(
            128, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        
        # Second hidden layer
        critic = nn.Dense(
            128, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        
        # Output layer - produces a single value
        critic = nn.Dense(
            1, 
            kernel_init=orthogonal(1.0), 
            bias_init=constant(0.0)
        )(critic)
        
        # Remove the unnecessary dimension
        value = jnp.squeeze(critic, axis=-1)
        
        return value

@flax.struct.dataclass
class PacknetState:
    '''
    Class to store the state of the Packnet
    '''
    masks: FrozenDict
    current_task: int
    train_mode: bool

class Packnet():
    '''
    Class that implements the Packnet CL-method
    '''
    def __init__(self, 
                 seq_length, 
                 prune_instructions=0.5, 
                 train_finetune_split=(1,1), 
                 prunable_layers=(nn.Conv, nn.Dense)
            ):
        '''
        Initializes the Packnet class
        @param seq_length: the length of the sequence
        @param prune_instructions: the percentage of the network to prune
        @param train_finetune_split: the split between training and finetuning
        @param prunable_layers: the layers that can be pruned
        '''
        self.seq_length = seq_length
        self.prune_instructions = prune_instructions
        self.train_finetune_split = train_finetune_split
        self.prunable_layers = prunable_layers
        
        # Create the pruning instructions based on the sequence length
        if seq_length: 
            self.create_pruning_instructions()

    def init_mask_tree(self, params):
        '''
        Initializes a pytree with a fixed size and shape to store all masks of previous tasks
        @param params: the parameters of the model, to get the shape of the masks
        Returns a mask Pytree of shape (seq_length, *params.shape) per leaf
        '''
        def make_mask_leaf(leaf):
            '''
            Initializes a mask for a single leaf
            @param leaf: the leaf of the pytree
            returns a mask that mirrors the parameter shape, but with a leading dimension of the number of tasks
            '''
            shape = (self.seq_length,) + leaf.shape
            return jnp.zeros(shape, dtype=bool)
        
        return jax.tree_util.tree_map(make_mask_leaf, params)
    
    def update_mask_tree(self, mask_tree, new_mask, current_task):
        '''
        Updates the mask tree with a new mask
        @param mask_tree: the current mask tree
        @param new_mask: the new mask to add
        returns the updated mask tree
        '''
        def update_mask_leaf(old_leaf, new_leaf):
            '''
            Updates a single leaf (a kernel or bias array of params) with a new mask
            @param mask: the current mask
            @param new_mask: the new mask to add
            returns the updated mask
            '''
            return old_leaf.at[current_task].set(new_leaf)

        return jax.tree_util.tree_map(update_mask_leaf, mask_tree, new_mask)

    def combine_masks(self, mask_tree, last_task):
        '''
        Combines the masks of all old tasks into a single mask to compare the current task against
        @param mask_tree: the mask tree
        returns the combined mask (mask with True for all fixed weights of previous tasks)
        '''
        def combine_masks_leaf(leaf):
            '''
            Combines the masks of all tasks for a single leaf (kernel or bias)
            @param leaf: the leaf of the mask tree
            returns the combined mask
            '''
            max_tasks = self.seq_length
            def combine_for_last_task(last_task):
                indices = jnp.arange(max_tasks)
                
                # Build a boolean mask where each element is True if its index is less than last_task
                prev_tasks = jax.lax.lt(indices, last_task) 
                prev_tasks = jax.lax.convert_element_type(prev_tasks, jnp.bool_)  
                
                # Reshape the prev_tasks mask to match the shape of the leaf
                new_shape = (max_tasks,) + (1,) * (leaf.ndim - 1) # (max_tasks, 1, 1, ...) 
                prev_tasks = jnp.reshape(prev_tasks, new_shape)
                
                # keep only the masks of the previous tasks, set the rest to all False
                masked = jnp.where(prev_tasks, leaf, False)
                
                # Combine the masks over all tasks 
                return jnp.any(masked, axis=0)
            
            return jax.lax.cond(last_task == 0,
                                lambda _: jnp.zeros(leaf.shape[1:], dtype=jnp.bool_),
                                combine_for_last_task,
                                last_task)
        return jax.tree_util.tree_map(combine_masks_leaf, mask_tree)
    
    def get_mask(self, mask_tree, task_id):
        '''
        returns the mask of a given task
        @param mask_tree: the mask tree
        @param task_id: the task id
        returns the mask of the given task
        '''
        def slice_mask_leaf(leaf):
            '''
            Slices the mask of a single leaf
            @param leaf: the leaf of the mask tree
            returns the mask of the given task
            '''
            return leaf[task_id]
        return jax.tree_util.tree_map(slice_mask_leaf, mask_tree)        

    def create_pruning_instructions(self):
        '''
        Creates the pruning instructions based on the sequence length
        '''
        assert self.seq_length is not None, "Sequence length not provided"

        if not isinstance(self.prune_instructions, list):
            # Check if the pruning instruction is a percentage 
            assert 0 < self.prune_instructions < 1, (
                "Pruning instructions should be a percentage"
                )
            # Create a list of pruning instructions for all tasks 
            self.prune_instructions = [self.prune_instructions] * (self.seq_length-1)

        assert len(self.prune_instructions) == self.seq_length-1, (
            "Must provide pruning instructions for each task"
            )
        
    def layer_is_prunable(self, layer_name):
        '''
        Checks if a layer is prunable
        @param layer_name: the name of the layer
        returns a boolean indicating whether the layer is prunable
        '''
        for prunable_type in self.prunable_layers:
            if prunable_type.__name__ in layer_name:
                return True
        return False
    
    def prune(self, params, prune_quantile, state: PacknetState):
        '''
        Prunes the model based on the pruning instructions
        @param model: the model to prune
        @param prune_quantile: the quantile to prune
        @param state: the packnet state
        returns the pruned model
        '''

        masks = jax.lax.cond(
            (state.current_task == 0) & (state.masks is None),
            lambda _: self.init_mask_tree(params),
            lambda _: state.masks,
            operand=None
        )

        state = state.replace(masks=masks)
        
        # Get the combined mask of all previous tasks
        combined_mask = self.combine_masks(state.masks, state.current_task)
        sparsity_mask = self.compute_sparsity(combined_mask)
        jax.debug.print("sparsity_mask: {sparsity_mask}", sparsity_mask=sparsity_mask)
        
        # Create a list for all prunable parameters
        all_prunable = jnp.array([])
        for layer_name, layer_dict in params.items():
            for param_name, param_array in layer_dict.items():
                if (self.layer_is_prunable(layer_name)) and ("bias" not in param_name):
                    # get the combined mask for this layer
                    prev_mask_leaf = combined_mask[layer_name][param_name]

                    # Get parameters not used by previous tasks
                    p = jnp.where(jnp.logical_not(prev_mask_leaf), param_array, jnp.nan)

                    # Concatenate with existing prunable parameters
                    if p.size > 0: 
                        all_prunable = jnp.concatenate([all_prunable.reshape(-1), p.reshape(-1)], axis=0)

        cutoff = jnp.nanquantile(jnp.abs(all_prunable), prune_quantile)
        # count the number of params under the cutoff
        num_pruned = jnp.sum(jnp.abs(all_prunable) < cutoff)
        jax.debug.print("num_pruned: {num_pruned}", num_pruned=num_pruned)
        mask = {}
        new_params = {}

        for layer_name, layer_dict in params.items():
            new_layer = {}
            mask_layer = {}
            for param_name, param_array in layer_dict.items():
                if (self.layer_is_prunable(layer_name)) and ("bias" not in param_name):
                    # get the params that are used by the previous tasks
                    prev_mask_leaf = combined_mask[layer_name][param_name] 

                    # Create new mask for the current parameter array
                    new_mask_leaf = jnp.logical_and(
                        jnp.abs(param_array) >= cutoff, 
                        jnp.logical_not(prev_mask_leaf)
                    )
                    # keep the fixed parameters and the parameters above the cutoff
                    complete_mask = jnp.logical_or(prev_mask_leaf, new_mask_leaf)

                    # Generate small random values instead of zeros
                    rng_key = jax.random.PRNGKey(state.current_task + 42)
                    rng_key = jax.random.fold_in(rng_key, hash(layer_name + param_name))
                    small_random_values = jax.random.normal(
                        rng_key, param_array.shape) * 0.001  # Small initialization
                
                    # prune the parameters
                    pruned_params = jnp.where(complete_mask, param_array, 0)
                    
                    mask_layer[param_name] = new_mask_leaf
                    new_layer[param_name] = pruned_params
                else:
                    mask_layer[param_name] = jnp.zeros(param_array.shape, dtype=bool)
                    new_layer[param_name] = param_array
            
            new_params[layer_name] = new_layer
            mask[layer_name] = mask_layer

        masks = self.update_mask_tree(state.masks, mask, state.current_task)
        state = state.replace(masks=masks)

        new_param_dict = new_params
        return new_param_dict, state

    def train_mask(self, state: PacknetState, train_state, params_copy): 
        '''
        Zeroes out the gradients of the fixed weights of previous tasks. 
        This mask should be applied after backpropagation and before each optimizer step during training
        '''

        # check if there are any masks to apply
        def first_task():
            # No previous tasks to fix - create a mask with the same process as combine_masks
            # but with all False values
            prev_mask = jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(x, dtype=bool), 
                train_state.params["params"]
            )
            return prev_mask
        
        def other_tasks():
            # get all weights allocated for previous tasks 
            prev_mask = self.combine_masks(state.masks, state.current_task)
            return prev_mask
        
        prev_mask = jax.lax.cond(
            state.current_task == 0,
            first_task,
            other_tasks,
        )

        def reset_params_train(param_leaf, param_copy_leaf, mask_leaf):
            """
            Resets the parameters to the old parameters if the parameter is fixed,
            to counteract the possible momentum that is still present in the update
            """
            # if the parameter is fixed (True), set it to the old parameter
            return jnp.where(mask_leaf, param_copy_leaf, param_leaf)
        
        # Extract the inner parameter dictionaries to match structures
        inner_params = train_state.params["params"]
        inner_params_copy = params_copy["params"]
        
        # apply the reset function to all parameters
        new_params = jax.tree_util.tree_map(reset_params_train, inner_params, inner_params_copy, prev_mask)

        return {"params": new_params}

    def fine_tune_mask(self, state: PacknetState, train_state, params_copy):
        '''
        Zeroes out the gradient of the pruned weights of the current task and previously fixed weights 
        This mask should be applied before each optimizer step during fine-tuning
        '''
        
        current_mask = self.get_mask(state.masks, state.current_task)

        def reset_params_finetune(param_leaf, param_copy_leaf, mask_leaf):
            """
            Resets the parameters to the old parameters if the parameter is fixed,
            to counteract the possible momentum that is still present in the update
            """
            # if the parameter is pruned (False), set it to the old parameter
            return jnp.where(mask_leaf, param_leaf, param_copy_leaf)
        
        # Extract the inner parameter dictionaries to match structures
        inner_params = train_state.params["params"]
        inner_params_copy = params_copy["params"]
        
        # apply the reset function to all parameters
        new_params = jax.tree_util.tree_map(reset_params_finetune, inner_params, inner_params_copy, current_mask)

        return {"params": new_params}

    def fix_biases(self, state: PacknetState):
        '''
        Set all masks for the biases to True after the first task,
        so that the biases will not be updated after the first task
        '''

        masks = state.masks
        def after_first_task(masks):
            # Iterate over all masks and set the biases to True
            for layer_name, layer_dict in masks.items():
                for param_name, mask_array in layer_dict.items():
                    if "bias" in param_name:
                        # Set the mask to True for all tasks
                        mask_array = jnp.ones(mask_array.shape, dtype=bool)
            return masks
        
        def first_task(masks):
            # No previous tasks to fix
            return masks
        
        masks =  jax.lax.cond(state.current_task == 0, first_task, after_first_task, masks)
        state = state.replace(masks=masks)

        return state

    def apply_eval_mask(self, params, task_id, state: PacknetState):
        '''
        Applies the mask of a given task to the model to revert to that network state
        '''
        assert len(state.masks) > task_id, "Current task index exceeds available masks"

        masked_params = {}

        # Iterate over prunable layers and collect the masks of previous tasks
        for layer_name, layer_dict in params.items():
            masked_layer_dict = {}
            for param_name, param_array in layer_dict.items():
                if self.layer_is_prunable and "bias" not in param_name:
                    full_param_name = f"{layer_name}/{param_name}"
                    prev_mask = jnp.zeros(param_array.shape, dtype=bool)
                    for i in range(0, task_id+1):
                        prev_mask = jnp.logical_or(prev_mask, state.masks[i][full_param_name])

                    # Zero out all weights that are not in the mask for this task
                    masked_layer_dict[param_name] = param_array * prev_mask
                else:
                    masked_layer_dict[param_name] = param_array
                
            masked_params[layer_name] = masked_layer_dict

        return masked_params                
        
    def mask_remaining_params(self, params, state: PacknetState):
        '''
        Masks the remaining parameters of the model that are not pruned
        typically called after the last task's initial training phase
        '''
        prev_mask = self.combine_masks(state.masks, state.current_task)

        mask = {}

        for layer_name, layer_dict in params.items():
            mask_layer = {}
            for param_name, param_array in layer_dict.items():
                if self.layer_is_prunable(layer_name) and "bias" not in param_name:

                    prev_mask_leaf = prev_mask[layer_name][param_name]
                    new_mask_leaf = jnp.logical_not(prev_mask_leaf)

                    mask_layer[param_name] = new_mask_leaf
                    
                else:
                    mask_layer[param_name] = jnp.zeros(param_array.shape, dtype=bool)

            mask[layer_name] = mask_layer

        masks = self.update_mask_tree(state.masks, mask, state.current_task)
        state = state.replace(masks=masks)

        # create the parameters to return the same shape as prune
        new_param_dict = params

        return new_param_dict, state

    def on_train_end(self, params, state: PacknetState):
        '''
        Handles the end of the training phase on a task
        '''
        # change the mode to finetuning
        state = state.replace(train_mode=False)

        prune_instructions = jnp.array(self.prune_instructions)

        def last_task(params):
            # if we are on the last task, mask all remaining parameters
            return self.mask_remaining_params(params, state)
             
        def other_tasks(params):
            # if we are not on the last task, prune the model
            prune_value = jnp.take(prune_instructions, state.current_task)
            return self.prune(params, prune_value, state)
            
        
        new_params, state = jax.lax.cond(
            state.current_task == self.seq_length-1,
            last_task,
            other_tasks,
            params
        )
        # fix the structure of the params:
        new_params = {"params": new_params}
        return new_params, state
    
    def on_finetune_end(self, state: PacknetState):
        '''
        Handles the end of the finetuning phase on a task
        '''
        state = state.replace(current_task=state.current_task+1, train_mode=True)
        
        # def first_task_biases(grads):
        #     return self.fix_biases(grads)
        
        # def other_tasks(grads):
        #     return grads
        
        # # apply the first task biases if we are on the first task
        # new_gradients = jax.lax.cond(
        #     state.current_task == 1,
        #     first_task_biases, 
        #     other_tasks, 
        #     grads
        #     )
        # new_gradients = {"params": new_gradients}
        return state

    def on_backwards_end(self, state: PacknetState, actor_train_state, params_copy):
        '''
        Handles the end of the backwards pass
        '''

        # fix the biases of the gradients
        state = self.fix_biases(state)

        def finetune(state):
            '''
            Revert the masked params to their original values
            '''
            return self.fine_tune_mask(state, actor_train_state, params_copy)
        
        def train(state):
            '''
            Revert the masked params to their original values
            '''
            return self.train_mask(state, actor_train_state, params_copy)   
        
        new_params = jax.lax.cond(
            state.train_mode, 
            train, 
            finetune,
            state
        )

        actor_train_state = actor_train_state.replace(params=new_params)
        return actor_train_state

    def get_total_epochs(self):
        return self.train_finetune_split[0] + self.train_finetune_split[1]
    
    def compute_sparsity(self, params):
        """Calculate percentage of zero weights in model"""
        total_params = 0
        zero_params = 0
        
        for layer_name, layer_dict in params.items():
            for param_name, param_array in layer_dict.items():
                if "kernel" in param_name:  # Only weight parameters
                    total_params += param_array.size
                    zero_params += jnp.sum(jnp.abs(param_array) < 1e-5)
        
        # print(f"Total params: {total_params}, Zero params: {zero_params}")
        
        sparsity = zero_params / total_params if total_params > 0 else 1
        sparsity = jnp.round(sparsity, 4)
        return sparsity



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

@dataclass
class Config:
    lr: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128
    total_timesteps: float = 6e6
    update_epochs: int = 8
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_shaping_horizon: float = 2.5e6
    activation: str = "tanh"
    env_name: str = "overcooked"
    alg_name: str = "ippo"

    # Packnet settings
    train_epochs: int = 8
    finetune_epochs: int = 2
    finetune_lr: float = 1e-4
    finetune_timesteps: int = 1e6

    seq_length: int = 4
    strategy: str = "random"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: ["asymm_advantages", "smallest_kitchen", "cramped_room", "easy_layout", "square_arena", "no_cooperation"])
    env_kwargs: Optional[Sequence[dict]] = None
    layout_name: Optional[Sequence[str]] = None
    log_interval: int = 50
    eval_num_steps: int = 1000 # number of steps to evaluate the model
    eval_num_episodes: int = 5 # number of episodes to evaluate the model
    
    anneal_lr: bool = False
    seed: int = 30
    num_seeds: int = 1
    
    # Wandb settings
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "ippo_continual"
    tags: List[str] = field(default_factory=list)

    # to be computed during runtime
    num_actors: int = 0
    num_updates: int = 0
    minibatch_size: int = 0
    finetune_updates: int = 0

    
############################
##### HELPER FUNCTIONS #####
############################

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

    
############################
##### MAIN FUNCTION    #####
############################


def main():
     # set the device to the first available GPU
    jax.config.update("jax_platform_name", "gpu")

    # print the device that is being used
    print("Device: ", jax.devices())
    
    config = tyro.cli(Config)

    # generate a sequence of tasks
    seq_length = config.seq_length
    strategy = config.strategy
    layouts = config.layouts
    config.env_kwargs, config.layout_name = generate_sequence(seq_length, strategy, layout_names=layouts, seed=config.seed)


    for layout_config in config.env_kwargs:
        layout_name = layout_config["layout"]
        layout_config["layout"] = overcooked_layouts[layout_name]
    
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    run_name = f'{config.alg_name}_Packnet_decoupled_seq{config.seq_length}_{config.strategy}_{timestamp}'
    exp_dir = os.path.join("runs", run_name)

    # Initialize WandB
    load_dotenv()
    wandb_tags = config.tags if config.tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project='Continual_IPPO', 
        config=config,
        sync_tensorboard=True,
        mode=config.wandb_mode,
        name=run_name,
        tags=wandb_tags,
    )

    # Set up Tensorboard
    writer = SummaryWriter(exp_dir)
    
    # add the hyperparameters to the tensorboard
    rows = []
    for key, value in vars(config).items():
        value_str = str(value).replace("\n", "<br>")
        value_str = value_str.replace("|", "\\|")  # escape pipe chars if needed
        rows.append(f"|{key}|{value_str}|")

    table_body = "\n".join(rows)
    markdown = f"|param|value|\n|-|-|\n{table_body}"
    writer.add_text("hyperparameters", markdown)

    def pad_observation_space():
        '''
        Function that pads the observation space of all environments to be the same size by adding extra walls to the outside.
        This way, the observation space of all environments is the same, and compatible with the network
        returns the padded environments
        '''
        envs = []
        for env_args in config.env_kwargs:
                # Create the environment
                env = make(config.env_name, **env_args)
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
    
    @partial(jax.jit, static_argnums=(1))
    def evaluate_model(actor_train_state, key):
        '''
        Evaluates the model by running 10 episodes on all environments and returns the average reward
        @param train_state: the current state of the training
        @param config: the configuration of the training
        returns the average reward
        '''

        def run_episode_while(env, key_r, actor_params, max_steps=400):
            """
            Run a single episode using jax.lax.while_loop 
            """
            class EvalState(NamedTuple):
                key: Any
                state: Any
                obs: Any
                done: bool
                total_reward: float
                step_count: int

            def cond_fun(state: EvalState):
                '''
                Checks if the episode is done or if the maximum number of steps has been reached
                @param state: the current state of the loop
                returns a boolean indicating whether the loop should continue
                '''
                return jnp.logical_and(jnp.logical_not(state.done), state.step_count < max_steps)

            def body_fun(state: EvalState):
                '''
                Performs a single step in the environment
                @param state: the current state of the loop
                returns the updated state
                '''
                # Unpack the state
                key, state_env, obs, _, total_reward, step_count = state

                # split the key into keys to sample actions and step the environment
                key, key_a0, key_a1, key_s = jax.random.split(key, 4)

                # Flatten observations
                flat_obs = {k: v.flatten() for k, v in obs.items()}

                def select_action(actor_train_state, rng, obs):
                    '''
                    Selects an action based on the policy network
                    @param params: the parameters of the network
                    @param rng: random number generator
                    @param obs: the observation
                    returns the action
                    '''
                    network_apply = actor_train_state.apply_fn
                    pi = network_apply(actor_params, obs)
                    return pi.sample(seed=rng)


                # Get action distributions
                action_a1 = select_action(actor_train_state, key_a0, flat_obs["agent_0"])
                action_a2 = select_action(actor_train_state, key_a1, flat_obs["agent_1"])

                # Sample actions
                actions = {
                    "agent_0": action_a1,
                    "agent_1": action_a2
                }

                # Environment step
                next_obs, next_state, reward, done_step, info = env.step(key_s, state_env, actions)
                done = done_step["__all__"]
                reward = reward["agent_0"]  
                total_reward += reward
                step_count += 1

                return EvalState(key, next_state, next_obs, done, total_reward, step_count)

            # Initialize the key and first state
            key, key_s = jax.random.split(key_r)
            obs, state = env.reset(key_s)
            init_state = EvalState(key, state, obs, False, 0.0, 0)

            # Run while loop
            final_state = jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body_fun,
                init_val=init_state
            )

            return final_state.total_reward

        # Loop through all environments
        all_avg_rewards = []

        envs = pad_observation_space()

        keys = jax.random.split(key, len(envs))

        for i, env in enumerate(envs):
            env = make(config.env_name, layout=env)  # Create the environment
            actor_params = actor_train_state.params
            env_key = keys[i]
            # Run k episodes
            all_rewards = jax.vmap(lambda k: run_episode_while(env, k, actor_params, config.eval_num_steps))(
                jax.random.split(env_key, config.eval_num_episodes)
            )
            
            avg_reward = jnp.mean(all_rewards)
            all_avg_rewards.append(avg_reward)

        return all_avg_rewards
    
    def get_rollout_for_visualization():
        '''
        Simulates the environment using the network
        @param train_state: the current state of the training
        @param config: the configuration of the training
        returns the state sequence
        '''

        # Add the padding
        envs = pad_observation_space()

        state_sequences = []
        for env_layout in envs:
            env = make(config.env_name, layout=env_layout)

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
        
    def visualize_environments():
        '''
        Visualizes the environments using the OvercookedVisualizer
        @param config: the configuration of the training
        returns None
        '''
        state_sequences = get_rollout_for_visualization()
        visualizer = OvercookedVisualizer()
        # animate all environments in the sequence
        for i, env in enumerate(state_sequences):
            visualizer.animate(state_seq=env, agent_view_size=5, filename=f"~/JAXOvercooked/environment_layouts/env_{config.layouts[i]}.gif")

        return None
    
    # padd all environments
    padded_envs = pad_observation_space()
    
    envs = []
    for env_layout in padded_envs:
        env = make(config.env_name, layout=env_layout)
        env = LogWrapper(env, replace_info=False)
        envs.append(env)


    # set extra config parameters based on the environment
    temp_env = envs[0]
    config.num_actors = temp_env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
    print(f"num_updates: {config.num_updates}")
    config.finetune_updates = config.finetune_timesteps // config.num_steps // config.num_envs
    print(f"finetune_updates: {config.finetune_updates}")
    config.minibatch_size = (config.num_actors * config.num_steps) // config.num_minibatches

    def linear_schedule(count):
        '''
        Linearly decays the learning rate depending on the number of minibatches and number of epochs
        returns the learning rate
        '''
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config.reward_shaping_horizon
    )


    actor = Actor(
        action_dim=temp_env.action_space().n, 
        activation=config.activation
    )

    critic = Critic(
        activation=config.activation
    )

     # Initialize the Packnet class
    packnet = Packnet(seq_length=config.seq_length, 
                      prune_instructions=0.1,
                      train_finetune_split=(config.train_epochs, config.finetune_epochs),
                      prunable_layers=[nn.Dense])
    
    
    # Initialize the network
    rng = jax.random.PRNGKey(config.seed)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)

    init_x = jnp.zeros(temp_env.observation_space().shape).flatten()
    actor_params = actor.init(actor_rng, init_x)
    critic_params = critic.init(critic_rng, init_x)

    # Initialize the optimizer
    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), 
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), 
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
    )

    # jit the apply function
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    # calculate sparsity
    sparsity = packnet.compute_sparsity(actor_params["params"])
    print(f"Sparsity: {sparsity}")


    # Initialize the Packnet state
    packnet_state = PacknetState(
        masks=packnet.init_mask_tree(actor_params["params"]),
        current_task=0,
        train_mode=True
    )

    # Initialize the training state      
    actor_train_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=actor_tx
    )
    critic_train_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=critic_tx
    )

    train_states = (actor_train_state, critic_train_state)

    # def get_shape(x):
    #     return x.shape if hasattr(x, "shape") else type(x)

    # # This returns a nested structure with each array replaced by its shape.
    # shapes = jax.tree_util.tree_map(get_shape, train_state.params)
    # print(shapes)

    @partial(jax.jit, static_argnums=(3))
    def train_on_environment(rng, train_states, packnet_state, env, env_counter):
        '''
        Trains the network using IPPO
        @param rng: random number generator 
        returns the runner state and the metrics
        '''
        print("Training on environment")

        actor_train_state, critic_train_state = train_states

        # reset the learning rate and the optimizer
        actor_tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm), 
            optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm), 
            optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
        )
        actor_train_state = actor_train_state.replace(tx=actor_tx)
        critic_train_state = critic_train_state.replace(tx=critic_tx)
        
        # Initialize and reset the environment 
        rng, env_rng = jax.random.split(rng) 
        reset_rng = jax.random.split(env_rng, config.num_envs) 
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng) 
        
        # TRAIN 
        def _update_step(runner_state, unused):
            '''
            perform a single update step in the training loop
            @param runner_state: the carry state that contains all important training information
            returns the updated runner state and the metrics 
            '''
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                '''
                selects an action based on the policy, calculates the log probability of the action, 
                and performs the selected action in the environment
                @param runner_state: the current state of the runner
                returns the updated runner state and the transition
                '''
                # Unpack the runner state
                train_states, env_state, packnet_state, last_obs, update_step, grads, rng = runner_state
                actor_train_state, critic_train_state = train_states
                # split the random number generator for action selection
                rng, _rng = jax.random.split(rng)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, env.agents, config.num_actors)
                # print("obs_shape", obs_batch.shape)
                
                # apply the policy network to the observations to get the suggested actions and their values
                pi = actor.apply(actor_train_state.params, obs_batch)
                value = critic.apply(critic_train_state.params, obs_batch)

                # sample the actions from the policy distribution 
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # format the actions to be compatible with the environment
                env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
                env_act = {k:v.flatten() for k,v in env_act.items()}
                
                # STEP ENV
                # split the random number generator for stepping the environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_envs)
                
                # simultaniously step all environments with the selected actions (parallelized over the number of environments with vmap)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                # REWARD SHAPING IN NEW VERSION
                
                # add the reward of one of the agents to the info dictionary
                info["reward"] = reward["agent_0"]

                current_timestep = update_step * config.num_steps * config.num_envs

                # add the shaped reward to the normal reward 
                reward = jax.tree_util.tree_map(lambda x,y: x+y * rew_shaping_anneal(current_timestep), reward, info["shaped_reward"])

                transition = Transition(
                    batchify(done, env.agents, config.num_actors).squeeze(), 
                    action,
                    value,
                    batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob,
                    obs_batch
                )

                runner_state = (train_states, env_state, packnet_state, obsv, update_step, grads, rng)
                return runner_state, (transition, info)
            
            # Apply the _env_step function a series of times, while keeping track of the runner state
            runner_state, (traj_batch, info) = jax.lax.scan(
                f=_env_step, 
                init=runner_state, 
                xs=None, 
                length=config.num_steps
            )  

            # unpack the runner state that is returned after the scan function
            train_states, env_state, packnet_state, last_obs, update_step, grads, rng = runner_state
            actor_train_state, critic_train_state = train_states
            # create a batch of the observations that is compatible with the network
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors)

            # apply the network to the batch of observations to get the value of the last state
            last_val = critic.apply(critic_train_state.params, last_obs_batch)
            
            # @profile
            def _calculate_gae(traj_batch, last_val):
                '''
                calculates the generalized advantage estimate (GAE) for the trajectory batch
                @param traj_batch: the trajectory batch
                @param last_val: the value of the last state
                returns the advantages and the targets
                '''
                def _get_advantages(gae_and_next_value, transition):
                    '''
                    calculates the advantage for a single transition
                    @param gae_and_next_value: the GAE and value of the next state
                    @param transition: the transition to calculate the advantage for
                    returns the updated GAE and the advantage
                    '''
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.gamma * next_value * (1 - done) - value # calculate the temporal difference
                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
                    ) # calculate the GAE (used instead of the standard advantage estimate in PPO)
                    
                    return (gae, value), gae
                
                # iteratively apply the _get_advantages function to calculate the advantage for each step in the trajectory batch
                _, advantages = jax.lax.scan(
                    f=_get_advantages,
                    init=(jnp.zeros_like(last_val), last_val),
                    xs=traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # calculate the generalized advantage estimate (GAE) for the trajectory batch
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            # @profile
            def _update_epoch(update_state, unused):
                '''
                performs a single update epoch in the training loop
                @param update_state: the current state of the update
                returns the updated update_state and the total loss
                '''
                
                def _update_minbatch(train_states, batch_info):
                    '''
                    performs a single update minibatch in the training loop
                    @param train_state: the current state of the training
                    @param batch_info: the information of the batch
                    returns the updated train_state and the total loss
                    '''
                    # unpack the parameters
                    actor_train_state, critic_train_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        '''
                        calculates the loss of the actor network
                        @param actor_params: the parameters of the actor network
                        @param traj_batch: the trajectory batch
                        @param gae: the generalized advantage estimate
                        returns the actor loss
                        '''
                        # Rerun the network
                        pi = actor.apply(actor_params, traj_batch.obs)
                        
                        # Calculate the log probability 
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor_unclipped = ratio * gae
                        loss_actor_clipped = (
                            jnp.clip(
                                ratio,
                                1.0 - config.clip_eps,
                                1.0 + config.clip_eps,
                            )
                            * gae
                        )

                        loss_actor = -jnp.minimum(loss_actor_clipped, loss_actor_unclipped)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config.clip_eps)
                        
                        actor_loss = (
                            loss_actor
                            - config.ent_coef * entropy
                        )
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)
                    
                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        '''
                        calculates the loss of the critic network
                        @param critic_params: the parameters of the critic network
                        @param traj_batch: the trajectory batch
                        @param targets: the targets
                        returns the critic loss
                        '''
                        # Rerun the network
                        value = critic.apply(critic_params, traj_batch.obs) 
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config.vf_coef * value_loss
                        return critic_loss, (value_loss)
                    
                    # returns a function with the same parameters as loss_fn that calculates the gradient of the loss function
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)

                    actor_loss, actor_grads = actor_grad_fn(actor_train_state.params, traj_batch, advantages)
                    critic_loss, critic_grads = critic_grad_fn(critic_train_state.params, traj_batch, targets)

                    # Create a copy of the parameters
                    actor_params_copy = actor_train_state.params.copy()

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    # Mask the gradients 
                    actor_train_state = packnet.on_backwards_end(packnet_state, actor_train_state, actor_params_copy)

                    del actor_params_copy

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_information = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                        "actor_grads": actor_grads,
                        "critic_grads": critic_grads,
                    }
                    
                    return (actor_train_state, critic_train_state), loss_information
                
                
                # unpack the update_state (because of the scan function)
                train_states, packnet_state, traj_batch, advantages, targets, rng = update_state
                
                # set the batch size and check if it is correct
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_actors
                ), "batch size must be equal to number of steps * number of actors"
                
                # create a batch of the trajectory, advantages, and targets
                batch = (traj_batch, advantages, targets)          

                # reshape the batch to be compatible with the network
                batch = jax.tree_util.tree_map(
                    f=(lambda x: x.reshape((batch_size,) + x.shape[2:])), tree=batch
                )
                # split the random number generator for shuffling the batch
                rng, _rng = jax.random.split(rng)

                # creates random sequences of numbers from 0 to batch_size, one for each vmap 
                permutation = jax.random.permutation(_rng, batch_size)

                # shuffle the batch
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                ) # outputs a tuple of the batch, advantages, and targets shuffled 

                minibatches = jax.tree_util.tree_map(
                    f=(lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:]))), tree=shuffled_batch,
                )

                train_states, loss_information = jax.lax.scan(
                    f=_update_minbatch, 
                    init=train_states,
                    xs=minibatches
                )
                
                update_state = (train_states, packnet_state, traj_batch, advantages, targets, rng)
                return update_state, loss_information

            # create a tuple to be passed into the jax.lax.scan function
            update_state = (train_states, packnet_state, traj_batch, advantages, targets, rng)

            update_state, loss_info = jax.lax.scan( 
                f=_update_epoch, 
                init=update_state, 
                xs=None, 
                length=config.update_epochs
            )

            # unpack update_state
            train_states, packnet_state, traj_batch, advantages, targets, rng = update_state

            # set the metric to be the information of the last update epoch
            metric = info

            # calculate the current timestep
            current_timestep = update_step*config.num_steps * config.num_envs
            update_step = update_step + 1
            
            def evaluate_and_log(rng, update_step, train_states):
                rng, eval_rng = jax.random.split(rng)
                # Unpack the train states
                actor_train_state, critic_train_state = train_states

                actor_train_state_eval = jax.tree_util.tree_map(lambda x: x.copy(), actor_train_state)
                grads_eval = jax.tree_util.tree_map(lambda x: x.copy(), loss_info["actor_grads"])

                def log_metrics(metric, update_step):
                     # average the metric
                    metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
                    sparsity_actor = packnet.compute_sparsity(actor_train_state_eval.params["params"])
                    sparsity_grads = packnet.compute_sparsity(grads_eval["params"])
                    # add the sparsity and mask compliance to the metric dictionary
                    metric["PackNet/sparsity_actor"] = sparsity_actor
                    metric["PackNet/sparsity_grads"] = sparsity_grads
                    metric["PackNet/current_task"] = packnet_state.current_task
                    metric["PackNet/train_mode"] = packnet_state.train_mode


                    # add the general metrics to the metric dictionary
                    metric["General/update_step"] = update_step
                    metric["General/env_step"] = update_step * config.num_steps * config.num_envs
                    metric["General/learning_rate"] = linear_schedule(update_step * config.num_minibatches * config.update_epochs)

                    # Losses section
                    metric["Losses/total_loss"] = loss_info["total_loss"].mean()
                    metric["Losses/value_loss"] = loss_info["value_loss"].mean()
                    metric["Losses/actor_loss"] = loss_info["actor_loss"].mean()
                    metric["Losses/entropy"] = loss_info["entropy"].mean()

                    # Rewards section
                    metric["General/shaped_reward_agent0"] = metric["shaped_reward"]["agent_0"]
                    metric["General/shaped_reward_agent1"] = metric["shaped_reward"]["agent_1"]
                    metric.pop("shaped_reward", None)
                    metric["General/shaped_reward_annealed_agent0"] = metric["General/shaped_reward_agent0"] * rew_shaping_anneal(current_timestep)
                    metric["General/shaped_reward_annealed_agent1"] = metric["General/shaped_reward_agent1"] * rew_shaping_anneal(current_timestep)

                    # Advantages and Targets section
                    metric["Advantage_Targets/advantages"] = advantages.mean()
                    metric["Advantage_Targets/targets"] = targets.mean()

                    # Evaluation section
                    for i in range(len(config.layout_name)):
                        metric[f"Evaluation/{config.layout_name[i]}"] = jnp.nan

                    evaluations = evaluate_model(actor_train_state_eval, eval_rng)
                    for i, evaluation in enumerate(evaluations):
                        metric[f"Evaluation/{config.layout_name[i]}"] = evaluation

                    # Extract parameters 
                    actor_params = jax.tree_util.tree_map(lambda x: x, actor_train_state_eval.params["params"])
                    actor_grads = jax.tree_util.tree_map(lambda x: x, grads_eval["params"])
                    
                    
                    def callback(args):
                        metric, update_step, env_counter, actor_params, actor_grads = args
                        update_step = int(update_step)
                        env_counter = int(env_counter)
                        real_step = (env_counter-1) * config.num_updates + update_step
                        for key, value in metric.items():
                            writer.add_scalar(key, value, real_step)
                        for layer, dict in actor_params.items():
                            for layer_name, param_array in dict.items():
                                writer.add_histogram(
                                    tag=f"weights/{layer}/{layer_name}", 
                                    values=jnp.array(param_array), 
                                    global_step=real_step,
                                    bins=100)
                                writer.add_histogram(
                                    tag=f"grads/{layer}/{layer_name}", 
                                    values=jnp.array(actor_grads[layer][layer_name]), 
                                    global_step=real_step,
                                    bins=100)

                    jax.experimental.io_callback(callback, None, (metric, update_step, env_counter, actor_params, actor_grads))
                    return None
                
                def do_not_log(metric, update_step):
                    return None
                
                jax.lax.cond((update_step % config.log_interval) == 0, log_metrics, do_not_log, metric, update_step)
            
            # Evaluate the model and log the metrics
            evaluate_and_log(rng=rng, update_step=update_step, train_states=train_states)

            # unpack the loss information
            actor_grads = loss_info["actor_grads"]
            actor_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=(0,1)), actor_grads)
            sparsity = packnet.compute_sparsity(actor_grads["params"])

            rng = update_state[-1]
            runner_state = (train_states, env_state, packnet_state, last_obs, update_step, actor_grads, rng)

            return runner_state, metric

        rng, train_rng = jax.random.split(rng)

        # initialize a carrier that keeps track of the states and observations of the agents
        actor_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), actor_train_state.params)
        runner_state = (train_states, env_state, packnet_state, obsv, 0, actor_grads, train_rng)
        
        # apply the _update_step function a series of times, while keeping track of the state 
        runner_state, metric = jax.lax.scan(
            f=_update_step, 
            init=runner_state, 
            xs=None, 
            length=config.num_updates
        )

        # unpack the runner state
        train_states, env_state, packnet_state, last_obs, update_step, actor_grads, rng = runner_state
        actor_train_state, critic_train_state = train_states

        # # Prune the model and update the parameters
        new_actor_params, packnet_state = packnet.on_train_end(actor_train_state.params["params"], packnet_state)
        
        # check the sparsity of the new params
        sparsity = packnet.compute_sparsity(new_actor_params["params"])
        jax.debug.print(
            "Sparsity after pruning: {sparsity}", sparsity=sparsity)
        
        # update the actor train state with the new parameters
        actor_train_state = actor_train_state.replace(params=new_actor_params)
        train_states = (actor_train_state, critic_train_state)

        rng, finetune_rng = jax.random.split(rng)

        # Create a new runner state for the finetuning phase
        runner_state = (train_states, env_state, packnet_state, last_obs, update_step, actor_grads, finetune_rng)

        runner_state, metric = jax.lax.scan(
            f=_update_step,
            init=runner_state, 
            xs=None, 
            length=config.finetune_updates
        )

        # check the sparsity after finetuning
        actor_train_state = runner_state[0][0]
        sparsity = packnet.compute_sparsity(actor_train_state.params["params"])
        jax.debug.print(
            "Sparsity after finetuning: {sparsity}", sparsity=sparsity)

        # handle the end of the finetune phase 
        packnet_state = packnet.on_finetune_end(packnet_state)

        # add the packnet_state to the new runner state
        runner_state = (train_states, env_state, packnet_state, last_obs, update_step, actor_grads, finetune_rng)

        return runner_state, metric

    def loop_over_envs(rng, train_states, envs, packnet_state):
        '''
        Loops over the environments and trains the network
        @param rng: random number generator
        @param train_state: the current state of the training
        @param envs: the environments
        returns the runner state and the metrics
        '''
        # split the random number generator for training on the environments
        rng, *env_rngs = jax.random.split(rng, len(envs)+1)

        # counter for the environment 
        env_counter = 1

        for env_rng, env in zip(env_rngs, envs):
            # Call the train_on_environment function - CHANGE THIS LINE:
            runner_state, metrics = train_on_environment(env_rng, train_states, packnet_state, env, env_counter)
            
            # unpack the runner state
            train_states, env_state, packnet_state, last_obs, update_step, grads, rng = runner_state

            # save the model
            path = f"checkpoints/overcooked/{run_name}/model_env_{env_counter}"
            save_params(path, train_states)

            # update the environment counter
            env_counter += 1

        return runner_state

    def save_params(path, train_states):
        '''
        Saves the parameters of the network
        @param path: the path to save the parameters
        @param train_state: the current state of the training
        returns None
        '''
        actor_train_state, critic_train_state = train_states
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    {"actor_params": actor_train_state.params,
                     "critic_params": critic_train_state.params},
                )
            )
        print('model saved to', path)
        
        
    # Run the model
    rng, train_rng = jax.random.split(rng)
    # apply the loop_over_envs function to the environments
    runner_state = loop_over_envs(train_rng, train_states, envs, packnet_state)
    

def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)


if __name__ == "__main__":
    print("Running main...")
    main()

