#!/usr/bin/env python
"""Continual‑learning PPO trainer for **one** chef.

* Works with the `OvercookedSingle` env from `overcooked_single_agent.py`.
* Supports the regular CL penalties EWC, MAS, L2 (imported from the same
  folder).
* No multi‑agent cruft: everything is batched over environments only.
* Intended for quick hacking rather than production—trim or extend as needed.

Usage
-----
```bash
python ppo_single_agent_cl.py --seq_length 4 --cl_method ewc --ewc_mode online
```
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, NamedTuple, Sequence, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from dotenv import load_dotenv
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter

from architectures.cnn import ActorCritic as CNNActorCritic  # noqa: E402
# Local imports --------------------------------------------------------------
from architectures.shared_mlp import ActorCritic as MLPActorCritic  # noqa: E402
from cl_methods.EWC import EWC  # noqa: E402
from cl_methods.L2 import L2  # noqa: E402
from cl_methods.MAS import MAS  # noqa: E402
from jax_marl import make
from jax_marl.environments.env_selection import generate_sequence  # noqa: E402
from jax_marl.environments.overcooked_environment.common import OBJECT_TO_INDEX
from jax_marl.environments.overcooked_environment.overcooked_single import OvercookedSingle
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer  # noqa: E402
from jax_marl.wrappers.baselines import LogWrapper


# ---------------------------------------------------------------------------

@dataclass
class Config:
    # General ----------------------------------------------------------------
    env_name: str = "overcooked_single"
    alg_name: str = "ppo"

    # PPO core ---------------------------------------------------------------
    lr: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128  # rollout length (T)
    total_timesteps: float = 1e7
    update_epochs: int = 8
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    explore_fraction: float = 0.0  # ratio of random‑policy timesteps per task

    # Net --------------------------------------------------------------------
    activation: str = "relu"
    use_cnn: bool = False
    big_network: bool = False
    use_layer_norm: bool = True
    use_task_id: bool = False
    use_multihead: bool = False
    shared_backbone: bool = False
    regularize_heads: bool = True

    # Continual learning -----------------------------------------------------
    cl_method: str = "ewc"  # ewc|mas|l2|none
    reg_coef: float = 1e7
    importance_episodes: int = 5
    importance_steps: int = 500
    ewc_mode: str = "online"  # online|last|multi
    ewc_decay: float = 0.9
    normalize_importance: bool = False
    regularize_critic: bool = False

    # Task sequence ----------------------------------------------------------
    seq_length: int = 2
    strategy: str = "random"
    layouts: Sequence[str] = field(default_factory=list)
    height_min: int = 5
    height_max: int = 10
    width_min: int = 5
    width_max: int = 10
    wall_density: float = 0.15

    # Misc -------------------------------------------------------------------
    anneal_lr: bool = False
    seed: int = 0
    log_interval: int = 50  # updates
    eval_episodes: int = 5
    eval_steps: int = 1000
    gif_len: int = 300
    record_gif: bool = True

    # WandB / TB -------------------------------------------------------------
    wandb_mode: str = "online"
    entity: str | None = None
    project: str = "COOX"
    tags: List[str] = field(default_factory=list)

    # Runtime‑filled ---------------------------------------------------------
    num_updates: int = 0
    minibatch_size: int = 0


# ---------------------------------------------------------------------------
# Helper: simple single‑agent rollout buffer (T * N batch dim first)
# ---------------------------------------------------------------------------
class RollBuf(NamedTuple):
    obs: chex.Array  # (T, N, obs_dim)
    action: chex.Array  # (T, N)
    logp: chex.Array  # (T, N)
    value: chex.Array  # (T, N)
    reward: chex.Array  # (T, N)
    done: chex.Array  # (T, N)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def linear_schedule(lr: float, num_updates: int):
    return optax.linear_schedule(lr, 0.0, num_updates)


# CL method map
METHODS = {
    "ewc": lambda cfg: EWC(mode=cfg.ewc_mode, decay=cfg.ewc_decay),
    "mas": lambda cfg: MAS(),
    "l2": lambda cfg: L2(),
    "none": lambda cfg: None,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = tyro.cli(Config)
    key = jax.random.PRNGKey(cfg.seed)

    # ------------------------------------------------------------------
    # Task sequence layouts
    env_kwargs, layout_names = generate_sequence(
        sequence_length=cfg.seq_length,
        strategy=cfg.strategy,
        layout_names=cfg.layouts,
        seed=cfg.seed,
        height_rng=(cfg.height_min, cfg.height_max),
        width_rng=(cfg.width_min, cfg.width_max),
        wall_density=cfg.wall_density,
    )
    layouts = [kw["layout"] for kw in env_kwargs]

    # Pad to max H/W so CNN sees consistent spatial dims ----------------
    max_h = max(l["height"] for l in layouts)
    max_w = max(l["width"] for l in layouts)

    def pad_layout(l, max_h, max_w):
        l = flax.core.unfreeze(l)
        dh, dw   = max_h - l["height"], max_w - l["width"]
        top, lef = dh // 2, dw // 2
        old_w    = l["width"]

        def shift(idxs):
            rows = idxs // old_w + top
            cols = idxs %  old_w + lef
            return rows * max_w + cols

        for k in ["wall_idx","agent_idx","goal_idx",
                  "plate_pile_idx","onion_pile_idx","pot_idx"]:
            l[k] = jnp.asarray([shift(i) for i in l[k]], dtype=jnp.uint32)

        # -------- new: close the padded frame with walls ------------------
        border = []
        for r in range(max_h):
            for c in range(max_w):
                if r < top or r >= top + l["height"] or c < lef or c >= lef + l["width"]:
                    border.append(r * max_w + c)
        l["wall_idx"] = jnp.concatenate([l["wall_idx"],
                                         jnp.asarray(border, dtype=jnp.uint32)])
        # ------------------------------------------------------------------
        l["height"], l["width"] = max_h, max_w
        return flax.core.freeze(l)

    layouts = [pad_layout(l, max_h, max_w) for l in layouts]

    # ------------------------------------------------------------------
    # Network
    obs_shape = (max_w, max_h, 26)
    obs_dim = int(np.prod(obs_shape))
    Net = CNNActorCritic if cfg.use_cnn else MLPActorCritic
    net = Net(len(OvercookedSingle().action_set), cfg.activation, cfg.seq_length, cfg.use_multihead,
              cfg.shared_backbone, cfg.big_network, cfg.use_task_id, cfg.regularize_heads, cfg.use_layer_norm)
    key, sub = jax.random.split(key)
    dummy = jnp.zeros((1, *obs_shape)) if cfg.use_cnn else jnp.zeros((1, obs_dim))
    params = net.init(sub, dummy)

    # ------------------------------------------------------------------
    # Optimiser & state
    cfg.num_updates = cfg.total_timesteps // cfg.num_steps // cfg.num_envs
    cfg.minibatch_size = cfg.num_steps // cfg.num_minibatches

    lr_sched = linear_schedule(cfg.lr, cfg.num_updates) if cfg.anneal_lr else cfg.lr
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr_sched, eps=1e-5))
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    # Continual‑learning helper
    cl_impl = METHODS[cfg.cl_method.lower()](cfg) if cfg.cl_method.lower() != "none" else None
    cl_state = cl_impl.init_state(params, cfg.regularize_critic, cfg.regularize_heads) if cl_impl else None

    # Logging ----------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    network = "shared_mlp" if cfg.shared_backbone else "mlp"
    run_name = f'{cfg.alg_name}_{cfg.cl_method}_{network}_seq{cfg.seq_length}_{cfg.strategy}_seed_{cfg.seed}_{timestamp}'
    exp_dir = os.path.join("runs", run_name)
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=cfg.project, config=vars(cfg), mode=cfg.wandb_mode, tags=cfg.tags, name=run_name)

    tb = SummaryWriter(exp_dir)

    # Visualiser for GIFs
    viz = OvercookedVisualizer(num_agents=1)

    # ------------------------------------------------------------------
    # Training loop over tasks -----------------------------------------
    # ------------------------------------------------------------------
    for task_idx, layout in enumerate(layouts):
        env = make(cfg.env_name, layout=layout)  # Create the environment
        env = LogWrapper(env, replace_info=False)
        key, sub = jax.random.split(key)
        train_state, cl_state = train_on_task(env, train_state, sub, cfg, net, cl_impl, cl_state, task_idx,
                                              layout_names[task_idx], viz, exp_dir)

    print("Training complete.")


# ---------------------------------------------------------------------------
# Per‑task PPO ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def train_on_task(env: OvercookedSingle, ts: TrainState, rng: chex.PRNGKey, cfg: Config,
                  net, cl_impl, cl_state, task_idx: int, layout_name: str, viz, exp_dir: str):
    """Run PPO on one environment."""

    obs_shape = env.observation_space().shape
    obs_dim = int(np.prod(obs_shape))
    key, sub = jax.random.split(rng)

    # Vectorised env state ------------------------------------------------
    keys_reset = jax.random.split(sub, cfg.num_envs)
    obs_dict, env_state = jax.vmap(env.reset)(keys_reset)
    # extract single-agent observations
    obs = obs_dict["agent_0"]

    # always keep values as (N, 1)
    def _ensure_col(x):
        return x[:, None] if x.ndim == 1 else x

    # Rollout & update ----------------------------------------------------
    def rollout_step(carry, _):
        key, env_state, obs = carry
        key, sub = jax.random.split(key)

        batch_obs = obs if cfg.use_cnn else jnp.reshape(obs, (cfg.num_envs, obs_dim))
        pi, v = net.apply(ts.params, batch_obs)
        v = _ensure_col(v)
        a = pi.sample(seed=sub)
        logp = pi.log_prob(a)

        # step the vectorised envs
        key_env = jax.random.split(sub, cfg.num_envs)
        actions = {"agent_0": a}
        nxt_obs_dict, nxt_state, reward, done, _ = jax.vmap(env.step)(key_env, env_state, actions)
        nxt_obs = nxt_obs_dict["agent_0"]  # <-- keep the same type (array)

        obs_flat = batch_obs  # <-- already (N, obs_dim) when not using CNN
        data = (obs_flat, a, logp, v,
                reward["agent_0"][:, None],
                done["__all__"])

        carry = (key, nxt_state, nxt_obs)  # <-- still (PRNGKey, state, array)
        return carry, data

    def one_update(ts: TrainState, carry_in: Tuple[chex.PRNGKey, Any, chex.Array]):
        """Collect rollout then optimise."""
        carry, traj = jax.lax.scan(rollout_step, carry_in, None, length=cfg.num_steps)
        key, env_state, obs = carry
        (obs_b, a_b, logp_b, v_b, r_b, d_b) = map(
            lambda x: x.reshape(cfg.num_steps, cfg.num_envs, -1) if x.ndim == 2 else x, traj)

        # Bootstrap value
        batch_obs = obs if cfg.use_cnn else jnp.reshape(obs, (cfg.num_envs, obs_dim))
        _, v_last = net.apply(ts.params, batch_obs)
        v_last = _ensure_col(v_last)

        # GAE
        def gae(carry, inp):
            adv, next_v = carry
            r, d, v = inp
            delta = r + cfg.gamma * next_v * (1 - d) - v
            adv = delta + cfg.gamma * cfg.gae_lambda * (1 - d) * adv
            return (adv, v), adv

        (_, _), adv = jax.lax.scan(gae, (jnp.zeros_like(v_last), v_last), (r_b, d_b, v_b), reverse=True)
        returns = adv + v_b

        # Flatten T,N dims
        def flat(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        obs_f = obs_b.reshape(-1, *obs_shape) if cfg.use_cnn else flat(obs_b)
        a_f, logp_f, adv_f, ret_f, v_f = map(flat, (a_b, logp_b, adv, returns, v_b))

        # Update ----------------------------------------------------------
        def loss_fn(params, obs, act, logp_old, adv, ret, v_old):
            pi, v = net.apply(params, obs)
            v = _ensure_col(v)
            logp = pi.log_prob(act)
            ratio = jnp.exp(logp - logp_old)
            clip = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            actor_loss = -jnp.mean(jnp.minimum(ratio * adv, clip * adv))
            v_clipped = v_old + (v - v_old).clip(-cfg.clip_eps, cfg.clip_eps)
            critic_loss = 0.5 * jnp.mean(jnp.maximum((v - ret) ** 2, (v_clipped - ret) ** 2))
            ent = jnp.mean(pi.entropy())
            total = actor_loss + cfg.vf_coef * critic_loss - cfg.ent_coef * ent
            if cl_impl:
                total += cl_impl.penalty(params, cl_state, cfg.reg_coef)
            return total, (actor_loss, critic_loss, ent)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        def epoch_step(opt_state, _):
            key, params = opt_state
            # Shuffle indices
            key, sub = jax.random.split(key)
            idx = jax.random.permutation(sub, obs_f.shape[0])
            for i in range(cfg.num_minibatches):
                sl = idx[i * cfg.minibatch_size:(i + 1) * cfg.minibatch_size]
                (loss_val, aux), grads = grad_fn(params, obs_f[sl], a_f[sl], logp_f[sl], adv_f[sl], ret_f[sl], v_f[sl])
                params = optax.apply_updates(params, ts.tx.update(grads, ts.opt_state, params)[0])
            return (key, params), loss_val

        (key, new_params), _ = jax.lax.scan(epoch_step, (key, ts.params), None, length=cfg.update_epochs)
        ts = ts.replace(params=new_params)
        return (key, env_state, obs), ts

    # Loop over updates ----------------------------------------------------
    def body(i, state):
        key, env_state, obs, ts = state
        (key, env_state, obs), ts = one_update(ts, (key, env_state, obs))
        return (key, env_state, obs, ts)

    key, env_state, obs, ts = jax.lax.fori_loop(
        0.0, cfg.num_updates,
        body,
        (key, env_state, obs, ts),
    )

    # CL importance --------------------------------------------------------
    if cl_impl:
        imp = cl_impl.compute_importance(ts.params, env, net, task_idx, key, cfg.use_cnn, cfg.importance_episodes,
                                         cfg.importance_steps, cfg.normalize_importance)
        cl_state = cl_impl.update_state(cl_state, ts.params, imp)

    # Record gif -----------------------------------------------------------
    if cfg.record_gif:
        states = record_gif(env, ts, net, key, cfg.gif_len)
        viz.animate(states, agent_view_size=5, task_idx=task_idx, task_name=layout_name, exp_dir=exp_dir)

    return ts, cl_state


# ---------------------------------------------------------------------------
# Aux util
# ---------------------------------------------------------------------------

_EMPTY = OBJECT_TO_INDEX["empty"]


def _to_vis_state(s):
    """
    Wrap a single-agent `State` so that the visualizer (which expects
    two agents) is satisfied.
    """
    return SimpleNamespace(
        maze_map=np.asarray(s.env_state.maze_map),
        agent_inv=np.asarray([s.env_state.agent_inv, _EMPTY], dtype=np.uint8),
        agent_dir_idx=np.asarray([s.env_state.agent_dir_idx, 0], dtype=np.uint8),
        agent_pos=np.asarray(s.env_state.agent_pos),
        agent_dir=np.asarray(s.env_state.agent_dir),
        _raw=s,                          # keep original if you ever need it
    )

def record_gif(env, ts, net, rng, max_steps):
    rng, sub = jax.random.split(rng)
    # reset returns ({agent_id: obs_array}, state)
    obs_dict, state = env.reset(sub)
    obs = obs_dict["agent_0"]

    frames = [SimpleNamespace(env_state=_to_vis_state(state))]
    for _ in range(max_steps):
        # flatten if not CNN, else keep shape
        if net.__class__.__name__.startswith("CNN"):
            batch_obs = obs[None, ...]           # add batch dim → (1, H, W, C)
        else:
            batch_obs = obs.reshape(1, -1)  # (1, obs_dim)

        pi, _ = net.apply(ts.params, batch_obs)
        act = int(pi.sample(seed=rng)[0])

        rng, sub = jax.random.split(rng)
        # step returns (obs_dict, state, rewards, done, info)
        obs_dict, state, _, done, _ = env.step(sub, state, {"agent_0": act})

        obs = obs_dict["agent_0"]
        frames.append(SimpleNamespace(env_state=_to_vis_state(state)))
        if done["__all__"]:
            break

    return frames


if __name__ == "__main__":
    main()
