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
    num_tasks: int = 1
    use_multihead: bool = False
    shared_backbone: bool = False
    big_network: bool = False

    def setup(self):
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        neurons = 256 if self.big_network else 128

        if self.shared_backbone:
            # New architecture: shared trunk and multihead logic
            self.fc1 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="common_dense1")
            self.fc2 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="common_dense2")
            self.fc3 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="common_dense3")
            self.critic_fc1 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                       name="critic_dense1")
            self.critic_fc2 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                       name="critic_dense2")
            self.critic_fc2 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                       name="critic_dense3")
            actor_head_size = self.action_dim * (self.num_tasks if self.use_multihead else 1)
            critic_head_size = 1 * (self.num_tasks if self.use_multihead else 1)
            self.actor_head = nn.Dense(actor_head_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
                                       name="actor_head")
            self.critic_head = nn.Dense(critic_head_size, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
                                        name="critic_head")
        else:
            # Separate trunk: each branch is its own network

            # Actor branch
            self.actor_dense1 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                         name="actor_dense1")
            self.actor_dense2 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                         name="actor_dense2")
            self.actor_dense3 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                         name="actor_dense3")
            # If using multihead, output dimension is action_dim*num_tasks.
            actor_out_dim = self.action_dim * self.num_tasks if self.use_multihead else self.action_dim
            self.actor_out = nn.Dense(actor_out_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
                                      name="actor_head")

            # Critic branch
            self.critic_dense1 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                          name="critic_dense1")
            self.critic_dense2 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                          name="critic_dense2")
            self.critic_dense3 = nn.Dense(neurons, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                                          name="critic_dense3")
            critic_out_dim = 1 * self.num_tasks if self.use_multihead else 1
            self.critic_out = nn.Dense(critic_out_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
                                       name="critic_head")

    def __call__(self, x, env_idx=0):

        if self.shared_backbone:
            x = self.fc1(x)
            x = self.activation_fn(x)
            x = self.fc2(x)
            x = self.activation_fn(x)
            if self.big_network:
                x = self.fc3(x)
                x = self.activation_fn(x)
            if self.use_multihead:
                # Concatenate and then select the correct head
                actor_concat = self.actor_head(x)
                actor_logits = choose_head(actor_concat, self.num_tasks, env_idx)
            else:
                actor_logits = self.actor_head(x)
            pi = distrax.Categorical(logits=actor_logits)
            v = self.critic_fc1(x)
            v = self.activation_fn(v)
            v = self.critic_fc2(v)
            v = self.activation_fn(v)
            if self.big_network:
                v = self.critic_fc3(v)
                v = self.activation_fn(v)
            if self.use_multihead:
                critic_concat = self.critic_head(v)
                v = choose_head(critic_concat, self.num_tasks, env_idx)
            else:
                v = self.critic_head(v)
            v = jnp.squeeze(v, axis=-1)
            return pi, v
        else:
            # Actor branch: separate trunk
            actor = self.actor_dense1(x)
            actor = self.activation_fn(actor)
            actor = self.actor_dense2(actor)
            actor = self.activation_fn(actor)
            if self.big_network:
                actor = self.actor_dense3(actor)
                actor = self.activation_fn(actor)
            actor_all = self.actor_out(actor)
            if self.use_multihead:
                actor_logits = choose_head(actor_all, self.num_tasks, env_idx)
            else:
                actor_logits = actor_all
            pi = distrax.Categorical(logits=actor_logits)

            # Critic branch: separate trunk
            critic = self.critic_dense1(x)
            critic = self.activation_fn(critic)
            critic = self.critic_dense2(critic)
            critic = self.activation_fn(critic)
            if self.big_network:
                critic = self.critic_dense3(critic)
                critic = self.activation_fn(critic)
            critic_all = self.critic_out(critic)
            if self.use_multihead:
                value = choose_head(critic_all, self.num_tasks, env_idx)
            else:
                value = critic_all
            value = jnp.squeeze(value, axis=-1)
            return pi, value


