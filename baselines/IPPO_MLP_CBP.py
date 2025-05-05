# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from datetime import datetime
import copy
import pickle
import math
import flax
import flax.linen as nn
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from typing import Sequence, NamedTuple, Any, Optional, List, Tuple
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper

from jax_marl.registration import make
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
from dotenv import load_dotenv
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# jax.config.update("jax_platform_name", "gpu")

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

# Continual Backprop modules
class CBPDense(nn.Module):
    """Dense layer *plus* CBP bookkeeping (utility, age, counter)."""
    features: int
    name: str
    eta: float = 0.0   # utility decay, off by default
    kernel_init: callable = orthogonal(np.sqrt(2))
    bias_init: callable = constant(0.0)
    activation: str = "tanh"

    def setup(self):
        self.dense = nn.Dense(self.features,
                              kernel_init=self.kernel_init,
                              bias_init=self.bias_init,
                              name=f"{self.name}_d")
        if self.activation == "relu":
            self.act_fn = nn.relu
        elif self.activation == "tanh":
            self.act_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation function {self.activation}")

    def __call__(self, x, next_kernel, train: bool):
        # Forward pass
        y = self.dense(x)
        h = self.act_fn(y)
        if train:
            # ------------ CBP utility + age update ------------
            util = self.variable("cbp", f"{self.name}_util",
                                 lambda: jnp.zeros((self.features,)))
            age = self.variable("cbp", f"{self.name}_age",
                                lambda: jnp.zeros((self.features,), jnp.int32))
            ctr = self.variable("cbp", f"{self.name}_ctr",
                                lambda: jnp.zeros(()))  # scalar float
            # contribution =  |h| * Σ|w_out|
            w_abs_sum = jnp.sum(jnp.abs(next_kernel), axis=1)  # (n_units,)  # Flax weights are saved as (in, out). PyTorch is (out, in).
            abs_neuron_output = jnp.abs(h)  
            contrib = jnp.mean(abs_neuron_output, axis=0) * w_abs_sum   # Mean over the batch (dim 0 of h)
            util.value = util.value * self.eta + (1 - self.eta) * contrib
            age.value = age.value + 1  # +1 every fwd pass
            # replacement counter (ctr) is *not* touched here (it is updated after the optimizer step)
        return h


# -----------------------------------------------------------
class ActorCritic(nn.Module):
    """Two-layer actor & critic with CBP-enabled hidden layers."""
    action_dim: int
    activation: str = "tanh"

    # ---- CBP hyper-params ----
    cbp_eta: float = 0.0

    def setup(self):
        # --------- ACTOR ----------
        self.a_fc1 = CBPDense(128, eta=self.cbp_eta, name="actor_fc1", activation=self.activation)
        self.a_fc2 = CBPDense(128, eta=self.cbp_eta, name="actor_fc2", activation=self.activation)
        self.actor_out = nn.Dense(self.action_dim,
                                  kernel_init=orthogonal(0.01),
                                  bias_init=constant(0.0),
                                  name="actor_out")
        # --------- CRITIC ----------
        self.c_fc1 = CBPDense(128, eta=self.cbp_eta, name="critic_fc1", activation=self.activation)
        self.c_fc2 = CBPDense(128, eta=self.cbp_eta, name="critic_fc2", activation=self.activation)
        self.critic_out = nn.Dense(1,
                                   kernel_init=orthogonal(1.0),
                                   bias_init=constant(0.0),
                                   name="critic_out")

    def __call__(self, x, *, train: bool):
        # ---------- actor ----------
        h1 = self.a_fc1(x, next_kernel=self.param("actor_fc2_d", "kernel"), train=train)
        h2 = self.a_fc2(h1, next_kernel=self.param("actor_out", "kernel"), train=train)
        logits = self.actor_out(h2)
        pi = distrax.Categorical(logits=logits)

        # ---------- critic ----------
        hc1 = self.c_fc1(x, next_kernel=self.param("critic_fc2_d", "kernel"), train=train)
        hc2 = self.c_fc2(hc1, next_kernel=self.param("critic_out", "kernel"), train=train)
        value = jnp.squeeze(self.critic_out(hc2), axis=-1)
        return pi, value


# -----------------------------------------------------------
def weight_reinit(key, shape):
    """Should be the same weight initializer used at model start (orthogonal(√2))."""
    return orthogonal(np.sqrt(2))(key, shape)


def cbp_step(
        params: FrozenDict,
        cbp_state: FrozenDict,
        *,
        rng: jax.random.PRNGKey,
        maturity: int,
        rho: float,   # replacement rate (0.0 - 1.0)
    ) -> Tuple[FrozenDict, FrozenDict, jax.random.PRNGKey]:
    """
    Pure JAX function: one CBP maintenance step.
    * increments replacement counter for every layer
    * performs (possibly zero) replacements based on counter
    """
    p, s = unfreeze(params), unfreeze(cbp_state)  # python dicts (outside jit)

    def _layer(layer_name, next_layer_name, rng):
        util = s[f"{layer_name}_util"]
        age = s[f"{layer_name}_age"]
        ctr = s[f"{layer_name}_ctr"]
        mature_mask = age > maturity
        n_eligible = int(jnp.sum(mature_mask))
        ctr += n_eligible * rho

        # number of whole neurons to replace
        n_rep = int(math.floor(float(ctr)))
        ctr -= float(n_rep)
        s[f"{layer_name}_ctr"] = ctr
        if n_rep == 0:
            return rng

        # indices of mature neurons sorted by utility (ascending)
        idxs = np.where(np.array(mature_mask))[0]
        util_mature = util[idxs]
        to_rep_local = util_mature.argsort()[:n_rep]
        to_rep = idxs[to_rep_local]

        W_in = p[layer_name + "_d"]["kernel"]  # (in, out)
        b_in = p[layer_name + "_d"]["bias"]  # (out,)
        W_out = p[next_layer_name]["kernel"]  # (out, next_out)

        # --- neuron re-initialisation ---
        rng, k_init = jax.random.split(rng)
        tmp_W = weight_reinit(k_init, W_in.shape)
        # copy only chosen columns from tmp_W
        W_in  = W_in.at[:, to_rep].set(tmp_W[:, to_rep])
        b_in  = b_in.at[to_rep].set(0.0)                       # bias → 0
        W_out = W_out.at[to_rep, :].set(0.0)                   # outgoing weights → 0

        # reset bookkeeping
        util = util.at[to_rep].set(0.0)
        age  = age.at[to_rep].set(0)

        # write back
        p[layer_name + "_d"]["kernel"] = W_in
        p[layer_name + "_d"]["bias"] = b_in
        p[next_layer_name]["kernel"] = W_out
        s[f"{layer_name}_util"] = util
        s[f"{layer_name}_age"] = age
        return rng

    rng = _layer("actor_fc1", "actor_fc2_d", rng)
    rng = _layer("actor_fc2", "actor_out", rng)
    rng = _layer("critic_fc1", "critic_fc2_d", rng)
    rng = _layer("critic_fc2", "critic_out", rng)

    return freeze(p), freeze(s), rng


# -----------------------------------------------------------
class TrainStateCBP(TrainState):
    """TrainState with an *extra* collection that stores CBP variables."""
    cbp_state: FrozenDict


# -----------------------------------------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


# -----------------------------------------------------------
@dataclass
class Config:
    # ---------- PPO ----------
    lr: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128
    total_timesteps: float = 8e6
    update_epochs: int = 8
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # ---------- CBP ----------
    cbp_replace_rate: float = 1e-4
    cbp_maturity: int = 10_000
    cbp_decay: float = 0.0
    # ---------- misc ----------
    reward_shaping_horizon: float = 2.5e6
    activation: str = "tanh"
    env_name: str = "overcooked"
    alg_name: str = "ippo_cbp"
    seq_length: int = 6
    strategy: str = "random"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda:
    ["asymm_advantages", "smallest_kitchen", "cramped_room",
     "easy_layout", "square_arena", "no_cooperation"])
    env_kwargs: Optional[Sequence[dict]] = None
    layout_name: Optional[Sequence[str]] = None
    log_interval: int = 75
    eval_num_steps: int = 1_000
    eval_num_episodes: int = 5
    anneal_lr: bool = False
    seed: int = 30
    wandb_mode: str = "online"
    project: str = "COOX"
    tags: List[str] = field(default_factory=list)
    # computed at runtime
    num_actors: int = 0
    num_updates: int = 0
    minibatch_size: int = 0


# -------------------------
# ----  H E L P E R S  ----
# -------------------------
def batchify(x: dict, agent_list, num_actors):  # (dict) → (A·E, obs)
    return jnp.stack([x[a] for a in agent_list]).reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def calculate_sparsity(params, threshold=1e-5):
    flat, _ = jax.tree_util.tree_flatten(params)
    all_w = jnp.concatenate([p.ravel() for p in flat])
    return 100 * jnp.mean((jnp.abs(all_w) < threshold).astype(jnp.float32))


# -------------------------------  M A I N  ---------------------------------
def main():
    cfg = Config()
    # ---------- env curriculum ----------
    cfg.env_kwargs, cfg.layout_name = generate_sequence(
        cfg.seq_length, cfg.strategy, cfg.layouts, seed=cfg.seed)
    # patch layouts to full Overcooked layouts
    for kw in cfg.env_kwargs:
        kw["layout"] = overcooked_layouts[kw["layout"]]

    run_name = f"{cfg.alg_name}_seq{cfg.seq_length}_{cfg.strategy}_{datetime.now():%Y%m%d_%H%M}"
    # ----------------  logging  ----------------
    wandb.login()
    wandb.init(project=cfg.project, name=run_name, config=cfg, mode=cfg.wandb_mode,
               tags=cfg.tags)
    writer = SummaryWriter(os.path.join("runs", run_name))
    # ---------- build *one* padded env to know obs-dim ----------
    from flax.core import Scope  # noqa
    temp_env = make(cfg.env_name, **cfg.env_kwargs[0])
    act_dim = temp_env.action_space().n
    obs_dim = int(np.prod(temp_env.observation_space().shape))

    # ---------- network ----------
    net = ActorCritic(action_dim=act_dim,
                      activation=cfg.activation,
                      cbp_eta=cfg.cbp_decay)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    variables = net.init(init_rng, jnp.zeros((obs_dim,)), train=True)
    params, cbp_state = variables.pop("params")

    # ---------- optimiser ----------
    def lr_schedule(step):
        frac = 1.0 - step / (cfg.num_updates * cfg.num_minibatches)
        return cfg.lr * frac

    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(lr_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5))

    train_state = TrainStateCBP.create(apply_fn=net.apply,
                                       params=freeze(params),
                                       tx=tx,
                                       cbp_state=freeze(cbp_state))

    # --------------------------------------------------------------------
    # ----------------------   T R A I N I N G   -------------------------
    # --------------------------------------------------------------------
    def apply_net(ts: TrainStateCBP, obs, *, train_flag, rng):
        vars_all = {"params": ts.params, **ts.cbp_state}
        (pi, val), mut = net.apply(vars_all, obs, rngs={"dropout": rng},
                                   train=train_flag, mutable=["cbp"])
        ts = ts.replace(cbp_state=freeze(mut["cbp"]))
        return ts, pi, val

    def loss_grad(ts: TrainStateCBP, batch, rng):
        ts2, pi, value = apply_net(ts, batch.obs, train_flag=True, rng=rng)
        # PPO losses
        logp = pi.log_prob(batch.action)
        ratio = jnp.exp(logp - batch.log_prob)
        adv = batch.adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss_pi = -jnp.mean(jnp.minimum(ratio * adv,
                                        jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv))
        v_clip = batch.value + (value - batch.value).clip(-cfg.clip_eps, cfg.clip_eps)
        v_loss = 0.5 * jnp.mean(jnp.maximum((value - batch.ret) ** 2,
                                            (v_clip - batch.ret) ** 2))
        ent = jnp.mean(pi.entropy())
        loss = loss_pi + cfg.vf_coef * v_loss - cfg.ent_coef * ent
        return loss, (ts2, loss_pi, v_loss, ent)

    # jit for speed (purely functional)
    loss_grad = jax.jit(jax.value_and_grad(loss_grad, has_aux=True), static_argnums=2)

    # ---------------- curriculum loop -----------------
    global_step = 0
    for env_idx, env_cfg in enumerate(cfg.env_kwargs, start=1):
        env = make(cfg.env_name, **env_cfg)
        env = LogWrapper(env, replace_info=False)

        cfg.num_actors = env.num_agents * cfg.num_envs
        cfg.num_updates = int(cfg.total_timesteps // (cfg.num_envs * cfg.num_steps))
        cfg.minibatch_size = cfg.num_actors * cfg.num_steps // cfg.num_minibatches

        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, cfg.num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_rng)

        # ------------- per-environment training loop -------------
        for upd in range(cfg.num_updates):
            traj = []

            # ---------- COLLECT ----------
            for _ in range(cfg.num_steps):
                rng, step_rng = jax.random.split(rng)
                ts, pi, v = apply_net(train_state, batchify(obs, env.agents,
                                                            cfg.num_actors),
                                      train_flag=True, rng=step_rng)
                act = pi.sample(seed=step_rng)
                logp = pi.log_prob(act)
                env_act = unbatchify(act, env.agents, cfg.num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}
                rng, step_rng = jax.random.split(rng)
                step_rngs = jax.random.split(step_rng, cfg.num_envs)
                obs2, env_state, rew, done, info = jax.vmap(env.step)(step_rngs,
                                                                      env_state,
                                                                      env_act)
                traj.append(Transition(batchify(done, env.agents, cfg.num_actors).squeeze(),
                                       act, v, batchify(rew, env.agents,
                                                        cfg.num_actors).squeeze(),
                                       logp, batchify(obs, env.agents,
                                                      cfg.num_actors)))
                obs = obs2

                train_state = ts  # cbp_state already updated

            # --------- ADV / RETURN (GAE) ----------
            last_obs_batch = batchify(obs, env.agents, cfg.num_actors)
            _, last_val = train_state.apply_fn(
                {"params": train_state.params, **train_state.cbp_state},
                last_obs_batch, train=False)
            traj = Transition(*map(jnp.stack, zip(*traj)))  # stack along time

            def _gae(carry, t):
                gae, next_val = carry
                d, v, r = t.done, t.value, t.reward
                delta = r + cfg.gamma * next_val * (1 - d) - v
                gae = delta + cfg.gamma * cfg.gae_lambda * (1 - d) * gae
                return (gae, v), gae

            (_, _), adv = jax.lax.scan(_gae,
                                       (jnp.zeros_like(last_val), last_val),
                                       traj[::-1])
            adv = adv[::-1]
            returns = adv + traj.value
            traj = traj._replace(adv=adv, ret=returns)

            # flatten (T,A) → (T·A)
            flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]),
                                          traj)

            # ---------- PPO UPDATES ----------
            for epoch in range(cfg.update_epochs):
                perm = jax.random.permutation(rng,
                                              train_state.params["actor_out"]["bias"].shape[0] * cfg.num_steps)
                perm = perm.reshape((cfg.num_minibatches, -1))
                for idx in perm:
                    mini = jax.tree_util.tree_map(lambda x: x[idx], flat)
                    rng, grad_rng = jax.random.split(rng)
                    (aux, grads), grads_val = loss_grad(train_state, mini, grad_rng)
                    ts2, l_pi, l_v, ent = aux
                    train_state = ts2.apply_gradients(grads=grads_val)

                    # ------ CBP maintenance (after optimiser step) ------
                    rng, cbp_rng = jax.random.split(rng)
                    new_params, new_cstate, cbp_rng = cbp_step(
                        unfreeze(train_state.params),
                        unfreeze(train_state.cbp_state),
                        rng=cbp_rng,
                        maturity=cfg.cbp_maturity,
                        rho=cfg.cbp_replace_rate)
                    train_state = train_state.replace(params=new_params,
                                                      cbp_state=new_cstate)

            global_step += cfg.num_envs * cfg.num_steps

            if upd % cfg.log_interval == 0:
                wandb.log({
                    "step": global_step,
                    "loss_pi": float(l_pi),
                    "loss_v": float(l_v),
                    "entropy": float(ent),
                    "sparsity(%)": float(calculate_sparsity(train_state.params))
                }, step=global_step)

        # ---------------- save after finishing curriculum env ----------------
        ckpt_dir = Path("checkpoints") / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / f"env{env_idx}.flax", "wb") as f:
            f.write(flax.serialization.to_bytes(
                {"params": train_state.params,
                 "cbp_state": train_state.cbp_state}))
        print(f"Saved checkpoint for env {env_idx}.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Device(s):", jax.devices())
    main()
