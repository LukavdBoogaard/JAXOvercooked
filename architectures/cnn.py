import os

import jax
import numpy as np

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Optional


# ───────────────────────────────── helper ────────────────────────────────────
def choose_head(tensor: jnp.ndarray, num_heads: int, env_idx: int) -> jnp.ndarray:
    """Select the slice that corresponds to *env_idx* out of *num_heads*.

    The input is a batched matrix shaped (B, H⋅num_heads). We reshape it to
    (B, num_heads, H) and pick the required head.
    """
    B, tot = tensor.shape
    base = tot // num_heads
    return tensor.reshape(B, num_heads, base)[:, env_idx, :]


# ─────────────────────────────── base conv block ─────────────────────────────
class CNN(nn.Module):
    name_prefix: str  # "shared", "actor", "critic" … for layer names
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(32, (5, 5), name=f"{self.name_prefix}_conv1",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), name=f"{self.name_prefix}_conv2",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), name=f"{self.name_prefix}_conv3",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, name=f"{self.name_prefix}_proj",
                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return act(x)


# ─────────────────────────────── Actor-Critic ────────────────────────────────
class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"
    # Continual-learning specific
    num_tasks: int = 1
    use_multihead: bool = False
    shared_backbone: bool = True
    big_network: bool = False
    use_task_id: bool = False
    regularize_heads: bool = True

    @nn.compact
    def __call__(self, obs, *, env_idx: int = 0, task_onehot: Optional[jnp.ndarray] = None):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # ─── choose encoder(s) ─────────────────────────────────────────
        if self.shared_backbone:
            trunk = CNN("shared", self.activation)(obs)
            actor_emb = critic_emb = trunk  # same features
        else:
            actor_emb = CNN("actor", self.activation)(obs)
            critic_emb = CNN("critic", self.activation)(obs)

        # ─── append task ID one‑hot after the CNN & projection ────────────
        if self.use_task_id:
            if task_onehot is None:
                # Create default one‑hot from env_idx
                idxs = jnp.full((actor_emb.shape[0],), env_idx)
                task_onehot = jax.nn.one_hot(idxs, num_classes=self.num_tasks)
            # Concatenate to both branches
            actor_emb = jnp.concatenate([actor_emb, task_onehot], axis=-1)
            critic_emb = jnp.concatenate([critic_emb, task_onehot], axis=-1)

        # ─── actor branch ─────────────────────────────────────────────
        a = nn.Dense(128, name="actor_dense1",
                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_emb)
        a = act_fn(a)
        logits_dim = self.action_dim * (self.num_tasks if self.use_multihead else 1)
        logits_all = nn.Dense(logits_dim, name="actor_head",
                              kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        logits = choose_head(logits_all, self.num_tasks, env_idx) if self.use_multihead else logits_all
        pi = distrax.Categorical(logits=logits)

        # ─── critic branch ────────────────────────────────────────────
        c = nn.Dense(128, name="critic_dense1",
                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic_emb)
        c = act_fn(c)
        vdim = 1 * (self.num_tasks if self.use_multihead else 1)
        v_all = nn.Dense(vdim, name="critic_head",
                         kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        v = choose_head(v_all, self.num_tasks, env_idx) if self.use_multihead else v_all
        v = jnp.squeeze(v, axis=-1)

        return pi, v
