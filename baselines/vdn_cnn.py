"""
Specific to this implementation: CNN network and Reward Shaping Annealing as per Overcooked paper.
"""

import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import tyro
from datetime import datetime
from tensorboardX import SummaryWriter

import flax
import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze, unfreeze
import flashbax as fbx
import wandb
from dotenv import load_dotenv

from jax_marl import make
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.wrappers.baselines import (
    LogWrapper,
    CTRolloutManager,
)
from jax_marl.environments.overcooked_environment import overcooked_layouts


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64
        )(x)
        x = activation(x)

        return x


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        embedding = CNN()(x)
        x = nn.Dense(self.hidden_size)(embedding)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    avail_actions: dict
    rewards: dict
    dones: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


@dataclass
class Config:
    total_timesteps: float = 5e6
    num_envs: int = 64
    num_steps: int = 16
    hidden_size: int = 512
    num_layers: int = 2
    norm_type: str = "layer_norm"
    norm_input: bool = False
    eps_start: float = 1.0
    eps_finish: float = 0.2
    eps_decay: float = 0.2
    max_grad_norm: int = 10
    num_minibatches: int = 16
    num_epochs: int = 4
    lr: float = 0.000075
    lr_linear_decay: bool = True
    lambda_: float = 0.5 
    gamma: float = 0.99
    tau: float = 1
    buffer_size: int = 1e5
    buffer_batch_size: int = 128
    learning_starts: int = 1e3
    target_update_interval: int = 10
    loss_type: str = "vdn"

    
    env_kwargs: Dict[str, str] = field(default_factory=lambda: {"layout": "counter_circuit"})
    rew_shaping_horizon: float = 2.5e6
    test_during_training: bool = True
    test_interval: float = 0.01  # as a fraction of updates (e.g., log every 5% of the training process)
    test_num_steps: int = 400
    test_num_envs: int = 512  # number of episodes to average over
    seed: int = 30

    # Sequence settings 
    seq_length: int = 6
    strategy: str = "random"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: ["asymm_advantages", "smallest_kitchen", "cramped_room", "easy_layout", "square_arena", "no_cooperation"])
    env_kwargs: Optional[Sequence[dict]] = None
    layout_name: Optional[Sequence[str]] = None

    # Evaluation settings
    log_interval: int = 75 # log every 200 calls to update step
    eval_num_steps: int = 1000 # number of steps to evaluate the model
    eval_num_episodes: int = 5 # number of episodes to evaluate the model

    # Wandb settings
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "ippo_continual"
    tags: List[str] = field(default_factory=list)
    wandb_log_all_seeds: bool = False
    env_name: str = "overcooked"
    alg_name: str = "vdn"
    network_name: str = "cnn"
    cl_method_name: str = "none"

    # To be computed during runtime
    num_updates: int = 0



###################################################
############### HELPER FUNCTIONS ##################
###################################################

def batchify(x: dict, env: LogWrapper):
    '''
    stack the observations of all agents into a single array
    @param x: the observations
    @param env: the environment
    returns the batchified observations
    '''
    return jnp.stack([x[agent] for agent in env.agents], axis=0)

def unbatchify(x: jnp.ndarray, env: LogWrapper):
    '''
    unstack the observations of all agents into a dictionary
    @param x: the batchified observations
    @param env: the environment
    returns the unbatchified observations
    '''
    return {agent: x[i] for i, agent in enumerate(env.agents)}    

def get_greedy_actions(q_vals, valid_actions):
    '''
    Get the greedy actions from the Q-values
    @param q_vals: the Q-values
    @param valid_actions: the valid actions
    returns the greedy actions
    '''
    unavail_actions = 1 - valid_actions
    q_vals = q_vals - (unavail_actions * 1e10) #subtract a large number from the Q-values of unavailable actions
    return jnp.argmax(q_vals, axis=-1) #returns the index of the best action (with the highest q-value)

def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
    '''
    Function that performs epsilon-greedy exploration
    @param rng: the random number generator
    @param q_vals: the Q-values
    @param eps: the epsilon value
    @param valid_actions: the valid actions
    returns the chosen actions
    '''
    rng_a, rng_e = jax.random.split(
        rng
    )  # a key for sampling random actions and one for picking

    greedy_actions = get_greedy_actions(q_vals, valid_actions)

    # pick random actions from the valid actions
    def get_random_actions(rng, val_action):
        return jax.random.choice(
            rng,
            jnp.arange(val_action.shape[-1]),
            p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
        )

    _rngs = jax.random.split(rng_a, valid_actions.shape[0]) #the first dimension is the number of agents
    random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

    chosen_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,  # pick the actions that should be random
        random_actions,
        greedy_actions,
    )
    return chosen_actions

###################################################
############### MAIN FUNCTION #####################
###################################################

def main(): 
    # set the device 
    jax.config.update("jax_platform_name", "gpu")
    print("device: ", jax.devices())

    # load the config
    config = tyro.cli(Config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f'{config.alg_name}_seq{config.seq_length}_{config.strategy}_{timestamp}'
    exp_dir = os.path.join("runs", run_name)

    # Initialize WandB
    load_dotenv()
    wandb_tags = config.tags if config.tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project='COOX_benchmark', 
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

    # generate the sequence of environments
    config.env_kwargs, config.layout_name = generate_sequence(
        sequence_length=config.seq_length, 
        strategy=config.strategy, 
        layout_names=config.layouts, 
        seed=config.seed
    )

    # Create the environments
    for layout_config in config.env_kwargs:
        layout_name = layout_config["layout"]
        layout_config["layout"] = overcooked_layouts[layout_name]

    padded_envs = pad_observation_space()
    envs = []
    for env_layout in padded_envs:
        env = make(config.env_name, layout=env_layout)
        env = LogWrapper(env, replace_info=False)
        envs.append(env)
    
    config.num_updates = (
        config.total_timesteps // config.num_steps // config.num_envs
    )

    eps_scheduler = optax.linear_schedule(
        config.eps_start,
        config.eps_finish,
        config.eps_decay * config.num_updates,
    )

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config.rew_shaping_horizon
    )

    # Initialize the network
    rng = jax.random.PRNGKey(config.seed)

    wrapped_env = CTRolloutManager(
        envs[0], batch_size=config.num_envs, preprocess_obs=False
    )
    
    test_env = CTRolloutManager(
        env, batch_size=config.test_num_envs, preprocess_obs=False
    )

    network = QNetwork(
        action_dim=wrapped_env.max_action_space,
        hidden_size=config.hidden_size,
    )

    rng, agent_rng = jax.random.split(rng)

    init_x = jnp.zeros((1, *env.observation_space().shape))
    network_params = network.init(agent_rng, init_x)

    lr_scheduler = optax.linear_schedule(
        config.lr,
        1e-10,
        (config.num_epochs) * config.num_updates,
    )

    lr = lr_scheduler if config.lr_linear_decay else config.lr

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.radam(learning_rate=lr),
    )

    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=network_params,
        target_network_params=network_params,
        tx=tx,
    )

    # Create the replay buffer
    buffer = fbx.make_flat_buffer(
        max_length=int(config.buffer_size),
        min_length=int(config.buffer_batch_size),
        sample_batch_size=int(config.buffer_batch_size),
        add_sequences=False,
        add_batch_size=int(config.num_envs * config.num_steps),
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    rng, init_rng = jax.random.split(rng)

    init_obs, init_env_state = wrapped_env.batch_reset(init_rng)
    init_actions = {
        agent: wrapped_env.batch_sample(init_rng, agent) for agent in env.agents
    }
    init_obs, _, init_rewards, init_dones, init_infos = wrapped_env.batch_step(
        init_rng, init_env_state, init_actions
    )
    init_avail_actions = wrapped_env.get_valid_actions(init_env_state.env_state)
    init_timestep = Timestep(
        obs=init_obs,
        actions=init_actions,
        avail_actions=init_avail_actions,
        rewards=init_rewards,
        dones=init_dones,
    )
    init_timestep_unbatched = jax.tree.map(
        lambda x: x[0], init_timestep
    )  # remove the NUM_ENV dim
    buffer_state = buffer.init(init_timestep_unbatched)

    def train_on_environment(rng, train_state, env, buffer_state):
        '''
        Trains the agent on a single environment
        @param rng: the random number generator
        @param train_state: the current training state
        @param env: the environment to train on
        @param buffer_state: the current buffer state
        returns the updated training state
        '''

        print("Training on environment", env)

        # reset the learning rate if needed
        lr = lr_scheduler if config.lr_linear_decay else config.lr
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.radam(learning_rate=lr),
        )
        train_state = train_state.replace(tx=tx)

        # reset the timesteps, updates and gradient steps
        train_state = train_state.replace(timesteps=0, n_updates=0, grad_steps=0)

        # Initialize and reset the environment
        rng, env_rng = jax.random.split(rng) 
        reset_rng = jax.random.split(env_rng, config.num_envs) 
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng) 

        def _update_step(runner_state, unused):

            train_state, buffer_state, expl_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                '''
                steps the environment for a single step
                @param carry: the current state of the environment
                returns the new state of the environment and the timestep
                '''
                last_obs, env_state, rng = carry

                rng, rng_action, rng_step = jax.random.split(rng, 3)

                # Compute Q-values for all agents
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.params,
                    batchify(last_obs, env), 
                )  # (num_agents, num_envs, num_actions)

                # retrieve the valid actions
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)

                # perform epsilon-greedy exploration
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_action, env.num_agents)
                new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions, env)
                )
                actions = unbatchify(new_action, env)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_step, env_state, actions
                )

                # add shaped reward
                shaped_reward = infos.pop("shaped_reward")
                shaped_reward["__all__"] = batchify(shaped_reward, env).sum(axis=0)
                rewards = jax.tree.map(
                    lambda x, y: x + y * rew_shaping_anneal(train_state.timesteps),
                    rewards,
                    shaped_reward,
                )

                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    avail_actions=avail_actions,
                    rewards=rewards,
                    dones=dones,
                )
                return (new_obs, new_env_state, rng), (timestep, infos)

            # step the env
            rng, _rng = jax.random.split(rng)
            carry, (timesteps, infos) = jax.lax.scan(
                f=_step_env,
                init=(*expl_state, _rng),
                xs=None,
                length=config.num_steps,
            )
            expl_state = carry[:2]

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config.num_steps * config.num_envs
            )  # update timesteps count

            # BUFFER UPDATE
            timesteps = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), timesteps
            )  # (num_envs*num_steps, ...)
            buffer_state = buffer.add(buffer_state, timesteps)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience

                q_next_target = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.target_network_params, batchify(minibatch.second.obs, env)
                )  # (num_agents, batch_size, ...)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)

                vdn_target = minibatch.first.rewards["__all__"] + (
                    1 - minibatch.first.dones["__all__"]
                ) * config.gamma * jnp.sum(
                    q_next_target, axis=0
                )  # sum over agents

                def _loss_fn(params):
                    q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                        params, batchify(minibatch.first.obs, env)
                    )  # (num_agents, batch_size, ...)

                    # get logits of the chosen actions
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        batchify(minibatch.first.actions, env)[..., jnp.newaxis],
                        axis=-1,
                    ).squeeze()  # (num_agents, batch_size, )

                    chosen_action_q_vals = jnp.sum(chosen_action_q_vals, axis=0)
                    loss = jnp.mean((chosen_action_q_vals - vdn_target) ** 2)

                    return loss, chosen_action_q_vals.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state.timesteps > config.learning_starts
            )
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config.num_epochs
                ),
                lambda train_state, rng: (
                    (train_state, rng),
                    (
                        jnp.zeros(config.num_epochs),
                        jnp.zeros(config.num_epochs),
                    ),
                ),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config.target_update_interval == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config.tau,
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            if config.test_during_training:
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates
                    % int(config.num_updates * config.test_interval)
                    == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            # report on wandb if required
            if config.wandb_mode != "disabled":

                def callback(metrics, seed):
                    if config.wandb_log_all_seeds:
                        metrics.update(
                            {
                                f"rng{int(seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, config.seed)

            runner_state = (train_state, buffer_state, expl_state, test_state, rng)

            return runner_state, None
        
        def get_greedy_metrics(rng, train_state):
            if not config.test_during_training:
                return None
            """Help function to test greedy policy during training"""

            def _greedy_env_step(step_state, unused):
                last_obs, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.params,
                    batchify(last_obs, env),  # (num_agents, num_envs, num_actions)
                )  # (num_agents, num_envs, num_actions)
                actions = jnp.argmax(q_vals, axis=-1)
                actions = unbatchify(actions, env)
                new_obs, new_env_state, rewards, dones, infos = test_env.batch_step(
                    rng_s, env_state, actions
                )
                step_state = (new_obs, new_env_state, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            rng, _rng = jax.random.split(rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step,
                (init_obs, env_state, _rng),
                None,
                config.test_num_steps,
            )
            metrics = {
                "returned_episode_returns": jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        infos["returned_episode_returns"],
                        jnp.nan,
                    )
                )
            }
            return metrics
        
        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        rng, _rng = jax.random.split(rng)
        obs, env_state = wrapped_env.batch_reset(_rng)
        expl_state = (obs, env_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, expl_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            f=_update_step, 
            init=runner_state, 
            xs=None, 
            length=config.num_updates
        )

        return runner_state, metrics
    
    def loop_over_envs(rng, train_state, envs, buffer_state):
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
            runner_state, metrics = train_on_environment(env_rng, train_state, env, buffer_state)

            # unpack the runner state
            train_state, buffer_state, expl_state, test_state, rng = runner_state

            # save the model
            path = f"checkpoints/overcooked/{run_name}/model_env_{env_counter}"
            save_params(path, train_state)

            # update the environment counter
            env_counter += 1

        return runner_state

    def save_params(path, train_state):
        '''
        Saves the parameters of the network
        @param path: the path to save the parameters
        @param train_state: the current state of the training
        returns None
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    {"params": train_state.params}
                )
            )
        print('model saved to', path)

    # train the network
    rng, train_rng = jax.random.split(rng)
    runner_state = loop_over_envs(train_rng, train_state, envs, buffer_state)
        

if __name__ == "__main__":
    main()