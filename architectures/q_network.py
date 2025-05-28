import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


class CNN(nn.Module):
    """Tiny 3‑layer CNN ➜ 64‑unit projection with optional LayerNorm."""

    # name_prefix: str  # "shared" | "actor" | "critic"
    activation: str = "relu"
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh

        def conv(name: str, x, kernel):
            x = nn.Conv(32, kernel, name=f"{name}",
                        kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = act(x)
            return x

        x = conv("conv1", x, (5, 5))
        x = conv("conv2", x, (3, 3))
        x = conv("conv3", x, (3, 3))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, name=f"proj",
                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        if self.use_layer_norm:
            x = nn.LayerNorm(name=f"proj_ln", epsilon=1e-5)(x)
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

