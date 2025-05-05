
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


def choose_head(out: jnp.ndarray, num_heads: int, env_idx: int) -> jnp.ndarray:
    # out has shape (batch, base_dim * num_heads)
    batch = out.shape[0]
    base_dim = out.shape[1] // num_heads
    out = out.reshape(batch, num_heads, base_dim)
    return out[:, env_idx, :]


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"
    use_multihead: bool = False
    num_tasks: int = 1

    def setup(self):
        # Shared trunk layers:
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.fc1 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="common_dense1")
        self.fc2 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="common_dense2")

        # Critic trunk (separate for critic)
        self.critic_fc1 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                   name="critic_dense1")
        self.critic_fc2 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                   name="critic_dense2")

        # Actor and critic heads
        actor_head_size = self.action_dim
        critic_head_size = 1
        if self.use_multihead:
            actor_head_size *= self.num_tasks
            critic_head_size *= self.num_tasks
        self.actor_head = nn.Dense(actor_head_size, kernel_init=orthogonal(0.01),
                                   bias_init=constant(0.0), name="actor_head")
        self.critic_head = nn.Dense(critic_head_size, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
                                    name="critic_head")

    def __call__(self, x, env_idx=0):
        # Shared trunk for both actor and critic
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.activation_fn(x)

        if self.use_multihead:
            # Produce a concatenated actor output, then splice:
            actor_concat = self.actor_head(x)  # shape: (batch, action_dim * num_tasks)
            actor_logits = choose_head(actor_concat, self.num_tasks, env_idx)
        else:
            actor_logits = self.actor_head(x)

        pi = distrax.Categorical(logits=actor_logits)

        # Critic trunk:
        v = self.critic_fc1(x)
        v = self.activation_fn(v)
        v = self.critic_fc2(v)
        v = self.activation_fn(v)
        if self.use_multihead:
            critic_concat = self.critic_head(v)  # shape: (batch, 1 * num_tasks)
            v = choose_head(critic_concat, self.num_tasks, env_idx)
        else:
            v = self.critic_head(v)

        v = jnp.squeeze(v, axis=-1)
        return pi, v