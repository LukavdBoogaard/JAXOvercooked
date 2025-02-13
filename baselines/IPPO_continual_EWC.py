import os
from datetime import datetime
from typing import Sequence, NamedTuple, Any, Optional, List

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dotenv import load_dotenv
from flax import struct
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper

from jax_marl.environments.env_selection import generate_sequence
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.registration import make
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
from jax_marl.wrappers.baselines import LogWrapper

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import wandb
from functools import partial
from dataclasses import dataclass, field
import tyro
from tensorboardX import SummaryWriter


# Enable compile logging if desired
# jax.log_compiles(True)


class ActorCritic(nn.Module):
    """
    Class to define the actor-critic networks used in IPPO. Each agent has its own actor-critic network
    """
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # ACTOR
        actor_mean = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        # CRITIC
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        value = jnp.squeeze(critic, axis=-1)
        return pi, value


class Transition(NamedTuple):
    """
    Named tuple to store the transition information
    """
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


@struct.dataclass
class EWCState:
    old_params: FrozenDict
    fisher: FrozenDict


def copy_params(params):
    return jax.tree_map(lambda x: x.copy(), params)


def init_cl_state(params: FrozenDict) -> EWCState:
    """
    Initialize old_params with the current parameters, fisher with zeros
    """
    old_params = copy_params(params)
    fisher = jax.tree_map(lambda x: jnp.zeros_like(x), old_params)
    return EWCState(old_params=old_params, fisher=fisher)


def compute_fisher(train_state, env, key, n_samples=256):
    """
    Approximate diagonal Fisher by sampling from the current policy.
    We'll gather log_prob grads and square them, then scale so the
    average magnitude of Fisher is ~1.0.
    """

    def single_sample_fisher(params, obs, act):
        """
        Returns the squared gradient (elementwise) of log_prob w.r.t. params
        using a diagonal Fisher approximation.
        """

        def log_prob_sum(params_):
            pi, _ = train_state.apply_fn(params_, obs)
            return pi.log_prob(act).sum()

        grad_log_p = jax.grad(log_prob_sum)(params)
        return jax.tree_map(lambda g: g ** 2, grad_log_p)

    # Initialize fisher_accum to zero
    fisher_accum_init = jax.tree_map(lambda x: jnp.zeros_like(x), train_state.params)

    def body_fn(carry, _):
        rng, fisher_accum = carry
        rng, rng_obs, rng_act = jax.random.split(rng, 3)

        obs, _ = env.reset(rng_obs)  # Single environment reset
        obs_batched = {k: v.flatten() for k, v in obs.items()}
        pi, _ = train_state.apply_fn(train_state.params, obs_batched["agent_0"])
        act = pi.sample(seed=rng_act)

        fisher_sample = single_sample_fisher(train_state.params, obs_batched["agent_0"], act)
        fisher_accum = jax.tree_map(lambda f, fs: f + fs, fisher_accum, fisher_sample)
        return (rng, fisher_accum), None

    (rng_final, fisher_accum), _ = jax.lax.scan(
        body_fn,
        init=(key, fisher_accum_init),
        xs=None,
        length=n_samples
    )

    # Average the Fisher across samples
    fisher_accum = jax.tree_map(lambda x: x / n_samples, fisher_accum)

    # -------------------------------------------------------------------
    # OPTIONAL NORMALIZATION STEP:
    # Compute the mean magnitude and normalize so that fisher_accum has "average" scale ~1
    # -------------------------------------------------------------------
    # Sum up all entries of fisher_accum:
    total_abs = jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(jnp.abs(x)),
        fisher_accum,
        0.0
    )
    # Count total number of parameters
    param_count = jax.tree_util.tree_reduce(
        lambda acc, x: acc + x.size,
        fisher_accum,
        0
    )
    # Mean absolute value across entire Fisher
    fisher_mean = total_abs / (param_count + 1e-8)

    # Rescale all entries so the mean is ~1
    def normalize(x):
        return x / (fisher_mean + 1e-8)

    fisher_accum = jax.tree_map(normalize, fisher_accum)
    # -------------------------------------------------------------------

    return fisher_accum


def compute_ewc_loss(params: FrozenDict, ewc_state: EWCState, ewc_coef: float):
    """
    Compute EWC penalty: 0.5 * ewc_coef * sum( fisher * (params - old_params)^2 )
    """

    def penalty(p, old_p, f):
        return f * (p - old_p) ** 2

    ewc_term_tree = jax.tree_map(lambda p, old_p, f: penalty(p, old_p, f),
                                 params, ewc_state.old_params, ewc_state.fisher)
    ewc_term = jax.tree_util.tree_reduce(lambda acc, x: acc + x.sum(), ewc_term_tree, 0.0)
    return 0.5 * ewc_coef * ewc_term


def update_ewc_state(new_params: FrozenDict, fisher: FrozenDict) -> EWCState:
    """
    After finishing a task, record the new params as old_params and store the fisher
    """
    return EWCState(
        old_params=copy_params(new_params),
        fisher=fisher
    )


@dataclass
class Config:
    reg_coef: float = 300.0
    lr: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128
    total_timesteps: float = 8e6
    update_epochs: int = 8
    num_minibatches: int = 8
    eval_freq: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_shaping_horizon: float = 2.5e6
    activation: str = "tanh"
    env_name: str = "overcooked"
    alg_name: str = "IPPO"

    seq_length: int = 3
    strategy: str = "random"
    layouts: Optional[Sequence[str]] = field(
        default_factory=lambda: [
            "asymm_advantages", "smallest_kitchen", "cramped_room",
            "easy_layout", "square_arena", "no_cooperation"
        ]
    )
    env_kwargs: Optional[Sequence[dict]] = None
    layout_name: Optional[Sequence[str]] = None

    anneal_lr: bool = False
    seed: int = 30
    num_seeds: int = 1

    # Wandb settings
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "ippo_continual"
    tags: List[str] = None

    # to be computed during runtime
    num_actors: int = 0
    num_updates: int = 0
    minibatch_size: int = 0


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def main():
    # set the device to the first available GPU
    jax.config.update("jax_platform_name", "gpu")

    print("Device: ", jax.devices())
    config = tyro.cli(Config)

    # Generate a sequence of tasks
    seq_length = config.seq_length
    strategy = config.strategy
    layouts = config.layouts
    config.env_kwargs, config.layout_name = generate_sequence(
        seq_length, strategy, layout_names=layouts, seed=config.seed
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f'{config.alg_name}_EWC_seq{config.seq_length}_{config.strategy}_{timestamp}'
    exp_dir = os.path.join("runs", run_name)

    for layout_config in config.env_kwargs:
        layout_name = layout_config["layout"]
        layout_config["layout"] = overcooked_layouts[layout_name]

    # Initialize WandB
    load_dotenv()
    wandb_tags = config.tags if config.tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project='Continual_IPPO',
        config=config,
        sync_tensorboard=True,
        mode=config.wandb_mode,
        tags=wandb_tags,
        group="EWC",
        name=run_name
    )

    # Set up Tensorboard
    writer = SummaryWriter(exp_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])
        ),
    )

    def pad_observation_space():
        envs = []
        for env_args in config.env_kwargs:
            env = make(config.env_name, **env_args)
            envs.append(env)

        max_width, max_height = 0, 0
        for env in envs:
            max_width = max(max_width, env.layout["width"])
            max_height = max(max_height, env.layout["height"])

        padded_envs = []
        for env in envs:
            env = unfreeze(env.layout)
            width_diff = max_width - env["width"]
            height_diff = max_height - env["height"]

            left = width_diff // 2
            right = width_diff - left
            top = height_diff // 2
            bottom = height_diff - top
            width = env["width"]

            def adjust_indices(indices):
                adjusted_indices = []
                for idx in indices:
                    row = idx // width
                    col = idx % width
                    new_row = row + top
                    new_col = col + left
                    new_idx = new_row * (width + left + right) + new_col
                    adjusted_indices.append(new_idx)
                return jnp.array(adjusted_indices)

            env["wall_idx"] = adjust_indices(env["wall_idx"])
            env["agent_idx"] = adjust_indices(env["agent_idx"])
            env["goal_idx"] = adjust_indices(env["goal_idx"])
            env["plate_pile_idx"] = adjust_indices(env["plate_pile_idx"])
            env["onion_pile_idx"] = adjust_indices(env["onion_pile_idx"])
            env["pot_idx"] = adjust_indices(env["pot_idx"])

            padded_wall_idx = list(env["wall_idx"])
            for y in range(top):
                for x in range(max_width):
                    padded_wall_idx.append(y * max_width + x)
            for y in range(max_height - bottom, max_height):
                for x in range(max_width):
                    padded_wall_idx.append(y * max_width + x)
            for y in range(top, max_height - bottom):
                for x in range(left):
                    padded_wall_idx.append(y * max_width + x)
                for x in range(max_width - right, max_width):
                    padded_wall_idx.append(y * max_width + x)

            env["wall_idx"] = jnp.array(padded_wall_idx)
            env["height"] = max_height
            env["width"] = max_width
            padded_envs.append(freeze(env))
        return padded_envs

    @partial(jax.jit, static_argnums=(1))
    def evaluate_model(train_state, network, key):
        def run_episode_while(env, key_r, network, network_params, max_steps=1000):
            class LoopState(NamedTuple):
                key: Any
                state: Any
                obs: Any
                done: bool
                total_reward: float
                step_count: int

            def loop_cond(state: LoopState):
                return jnp.logical_and(~state.done, state.step_count < max_steps)

            def loop_body(state: LoopState):
                key, state_env, obs, _, total_reward, step_count = state
                key, key_a0, key_a1, key_s = jax.random.split(key, 4)
                flat_obs = {k: v.flatten() for k, v in obs.items()}

                def select_action(train_state, rng, obs):
                    network_apply = train_state.apply_fn
                    params = train_state.params
                    pi, value = network_apply(params, obs)
                    return pi.sample(seed=rng), value

                action_a1, _ = select_action(train_state, key_a0, flat_obs["agent_0"])
                action_a2, _ = select_action(train_state, key_a1, flat_obs["agent_1"])

                actions = {
                    "agent_0": action_a1,
                    "agent_1": action_a2
                }

                next_obs, next_state, reward, done_step, info = env.step(key_s, state_env, actions)
                done = done_step["__all__"]
                reward = reward["agent_0"]
                total_reward += reward
                step_count += 1
                return LoopState(key, next_state, next_obs, done, total_reward, step_count)

            key, key_s = jax.random.split(key_r)
            obs, state = env.reset(key_s)
            init_state = LoopState(key, state, obs, False, 0.0, 0)
            final_state = jax.lax.while_loop(loop_cond, loop_body, init_state)
            return final_state.total_reward

        all_avg_rewards = []
        envs = pad_observation_space()
        for env in envs:
            env = make(config.env_name, layout=env)
            network_params = train_state.params
            all_rewards = jax.vmap(lambda k: run_episode_while(env, k, network, network_params, 500))(
                jax.random.split(key, 5)
            )
            avg_reward = jnp.mean(all_rewards)
            all_avg_rewards.append(avg_reward)
        return all_avg_rewards

    padded_envs = pad_observation_space()
    envs = []
    for i, env_layout in enumerate(padded_envs):
        env = make(config.env_name, layout=env_layout, layout_name=config.layouts[i])
        env = LogWrapper(env, replace_info=False)
        envs.append(env)

    temp_env = envs[0]
    config.num_actors = temp_env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
    config.minibatch_size = (config.num_actors * config.num_steps) // config.num_minibatches

    def linear_schedule(count):
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config.total_timesteps
    )

    network = ActorCritic(temp_env.action_space().n, activation=config.activation)

    rng = jax.random.PRNGKey(config.seed)
    rng, network_rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space().shape).flatten()
    network_params = network.init(network_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
    )

    network.apply = jax.jit(network.apply)
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx
    )

    @partial(jax.jit, static_argnums=(3,))
    def train_on_environment(rng, train_state, ewc_state, env_idx):
        env = envs[env_idx]
        print(f"Training on environment {env_idx}: {env.name}")
        tx_local = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
        )
        train_state = train_state.replace(tx=tx_local)

        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_st, env_st, last_obs, update_step, rng_ = runner_state
                rng_, key_act = jax.random.split(rng_)

                obs_batch = batchify(last_obs, env.agents, config.num_actors)
                pi, value = network.apply(train_st.params, obs_batch)

                action = pi.sample(seed=key_act)
                log_prob = pi.log_prob(action)

                env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng_, key_step = jax.random.split(rng_)
                rng_step = jax.random.split(key_step, config.num_envs)
                obsv_, env_st_, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_st, env_act
                )
                info["reward"] = reward["agent_0"]
                current_timestep = update_step * config.num_steps * config.num_envs
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * rew_shaping_anneal(current_timestep),
                    reward, info["shaped_reward"]
                )

                transition = Transition(
                    batchify(done, env.agents, config.num_actors).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob,
                    obs_batch
                )
                return (train_st, env_st_, obsv_, update_step, rng_), (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                _env_step, runner_state, None, length=config.num_steps
            )
            train_st, env_st, last_obs, update_step, rng_ = runner_state

            last_obs_batch = batchify(last_obs, env.agents, config.num_actors)
            _, last_val = network.apply(train_st.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(carry, transition):
                    gae, next_value = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config.gamma * next_value * (1 - done) - value
                    gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    init=(jnp.zeros_like(last_val), last_val),
                    xs=traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_st, batch_info):
                    traj_b, gae, tgt = batch_info

                    def _loss_fn(params, ttraj, tgae, ttgt):
                        pi_, val_ = network.apply(params, ttraj.obs)
                        log_prob_ = pi_.log_prob(ttraj.action)

                        value_pred_clipped = ttraj.value + (val_ - ttraj.value).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(val_ - ttgt)
                        value_losses_clipped = jnp.square(value_pred_clipped - ttgt)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob_ - ttraj.log_prob)
                        tgae = (tgae - tgae.mean()) / (tgae.std() + 1e-8)
                        loss_actor_unclipped = ratio * tgae
                        loss_actor_clipped = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * tgae
                        loss_actor = -jnp.minimum(loss_actor_unclipped, loss_actor_clipped).mean()

                        entropy_ = pi_.entropy().mean()

                        # EWC penalty
                        ewc_penalty = compute_ewc_loss(params, ewc_state, config.reg_coef)

                        total_loss = (loss_actor
                                      + config.vf_coef * value_loss
                                      - config.ent_coef * entropy_
                                      + ewc_penalty)
                        return total_loss, (value_loss, loss_actor, entropy_, ewc_penalty)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, (v_loss, a_loss, ent, reg_loss)), grads_ = grad_fn(train_st.params, traj_b, gae, tgt)
                    train_st = train_st.apply_gradients(grads=grads_)
                    return train_st, (total_loss, v_loss, a_loss, ent, reg_loss)

                train_st, traj_b_, gae_, tgt_, rng_ = update_state

                batch_size = config.minibatch_size * config.num_minibatches
                assert batch_size == config.num_steps * config.num_actors, (
                    "batch size must be equal to num_steps * num_actors"
                )

                full_batch = (traj_b_, gae_, tgt_)
                full_batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), full_batch
                )
                rng_, key_perm = jax.random.split(rng_)
                permutation = jax.random.permutation(key_perm, batch_size)
                shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), full_batch)

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled
                )

                train_st, loss_info_ = jax.lax.scan(_update_minbatch, train_st, minibatches)
                total_loss_, v_loss_, a_loss_, ent_, reg_loss_ = loss_info_
                update_state = (train_st, traj_b_, gae_, tgt_, rng_)
                return update_state, (total_loss_, v_loss_, a_loss_, ent_, reg_loss_)

            update_state = (train_st, traj_batch, advantages, targets, rng_)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, length=config.update_epochs)
            train_st, traj_batch, advantages, targets, rng_ = update_state

            metric = info
            current_timestep = update_step * config.num_steps * config.num_envs
            metric = jax.tree_map(lambda x: x.mean(), metric)

            update_step += 1
            metric["General/update_step"] = update_step
            metric["General/env_step"] = update_step * config.num_steps * config.num_envs
            metric["General/learning_rate"] = linear_schedule(
                update_step * config.num_minibatches * config.update_epochs)

            total_loss_, v_loss_, a_loss_, ent_, reg_ = loss_info
            metric["Losses/total_loss"] = total_loss_.mean()
            metric["Losses/total_loss_max"] = total_loss_.max()
            metric["Losses/total_loss_min"] = total_loss_.min()
            metric["Losses/total_loss_var"] = total_loss_.var()
            metric["Losses/value_loss"] = v_loss_.mean()
            metric["Losses/actor_loss"] = a_loss_.mean()
            metric["Losses/entropy"] = ent_.mean()
            metric["Losses/reg_loss"] = reg_.mean()

            metric["General/shaped_reward_agent0"] = metric["shaped_reward"]["agent_0"]
            metric["General/shaped_reward_agent1"] = metric["shaped_reward"]["agent_1"]
            metric["General/shaped_reward_annealed_agent0"] = metric[
                                                                  "General/shaped_reward_agent0"] * rew_shaping_anneal(
                current_timestep)
            metric["General/shaped_reward_annealed_agent1"] = metric[
                                                                  "General/shaped_reward_agent1"] * rew_shaping_anneal(
                current_timestep)

            metric["Advantage_Targets/advantages"] = advantages.mean()
            metric["Advantage_Targets/targets"] = targets.mean()

            for i_ in range(len(config.layout_name)):
                metric[f"Evaluation/{config.layout_name[i_]}"] = jnp.nan

            rng_, eval_rng = jax.random.split(rng_)
            train_state_eval = jax.tree_map(lambda x: x.copy(), train_st)

            def true_fun(met):
                evaluations = evaluate_model(train_state_eval, network, eval_rng)
                for i_, evaluation in enumerate(evaluations):
                    met[f"Evaluation/{config.layout_name[i_]}"] = evaluation
                return met

            def false_fun(met):
                return met

            metric = jax.lax.cond((update_step % config.eval_freq) == 0, true_fun, false_fun, metric)

            def callback(met, update_step_):
                wandb.log(met)

            jax.debug.callback(callback, metric, update_step)

            runner_state_out = (train_st, env_st, last_obs, update_step, rng_)
            return runner_state_out, metric

        rng, train_rng_ = jax.random.split(rng)
        runner_state_init = (train_state, env_state, obsv, 0, train_rng_)

        runner_state_final, metric_ = jax.lax.scan(
            _update_step,
            init=runner_state_init,
            xs=None,
            length=config.num_updates
        )
        return runner_state_final

    def loop_over_envs(rng, train_state, ewc_state, envs):
        rngs = jax.random.split(rng, len(envs) + 1)
        main_rng = rngs[0]
        sub_rngs = rngs[1:]
        visualizer = OvercookedVisualizer()

        runner_state = None
        for i, (r, _) in enumerate(zip(sub_rngs, envs)):
            # --- Train on environment i using the *current* ewc_state ---
            runner_state = train_on_environment(r, train_state, ewc_state, i)
            train_state = runner_state[0]

            # --- Compute new Fisher, then update ewc_state for next tasks ---
            fisher = compute_fisher(train_state, envs[i], r, n_samples=256)
            ewc_state = update_ewc_state(train_state.params, fisher)

            # Generate & log a GIF after finishing task i
            states = record_gif_of_episode(train_state, envs[i], network)
            visualizer.animate(states, agent_view_size=5, task_idx=i, task_name=envs[i].name, exp_dir=exp_dir)

        return runner_state

    rng, train_rng = jax.random.split(rng)
    ewc_state = init_cl_state(train_state.params)

    loop_over_envs(train_rng, train_state, ewc_state, envs)


def sample_discrete_action(key, action_space):
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)


def record_gif_of_episode(train_state, env, network, max_steps=300):
    rng = jax.random.PRNGKey(0)
    rng, env_rng = jax.random.split(rng)
    obs, state = env.reset(env_rng)
    done = False
    step_count = 0
    states = [state]

    while not done and step_count < max_steps:
        flat_obs = {k: v.flatten() for k, v in obs.items()}
        act_keys = jax.random.split(rng, env.num_agents)
        actions = {}
        for i, agent_id in enumerate(env.agents):
            pi, _ = network.apply(train_state.params, flat_obs[agent_id])
            actions[agent_id] = pi.sample(seed=act_keys[i])

        rng, key_step = jax.random.split(rng)
        next_obs, next_state, reward, done_info, info = env.step(key_step, state, actions)
        done = done_info["__all__"]
        obs, state = next_obs, next_state
        step_count += 1
        states.append(state)

    return states


if __name__ == "__main__":
    print("Running main...")
    main()
