import distrax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class ActorCritic(nn.Module):
    """Two-layer actor & critic with CBP-enabled hidden layers."""
    action_dim: int
    activation: str = "tanh"
    cbp_eta: float = 0.0  # CBP utility decay, off by default

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

    def _maybe_kernel(self, module_name: str, shape):
        """Return real kernel if it exists, else zeros (during init)."""
        if self.has_variable("params", module_name):
            return self.scope.get_variable("params", module_name)["kernel"]
        return jnp.zeros(shape, dtype=jnp.float32)

    def __call__(self, x, *, train: bool):
        # shapes used only for the dummy kernel during init
        dummy_hid = (128, 128)
        dummy_out = (128, self.action_dim)

        # ----- get weights (kernels) of next_layer -------
        k_a2 = self._maybe_kernel("actor_fc2_d", dummy_hid)
        k_aout = self._maybe_kernel("actor_out", dummy_out)
        k_c2 = self._maybe_kernel("critic_fc2_d", dummy_hid)
        k_cout = self._maybe_kernel("critic_out", (128, 1))

        # ---------- actor ----------
        h1 = self.a_fc1(x, next_kernel=k_a2, train=train)
        h2 = self.a_fc2(h1, next_kernel=k_aout, train=train)
        logits = self.actor_out(h2)
        pi = distrax.Categorical(logits=logits)

        # ---------- critic ----------
        hc1 = self.c_fc1(x, next_kernel=k_c2, train=train)
        hc2 = self.c_fc2(hc1, next_kernel=k_cout, train=train)
        value = jnp.squeeze(self.critic_out(hc2), axis=-1)
        return pi, value


class CBPDense(nn.Module):
    """Dense layer *plus* CBP bookkeeping (utility, age, counter)."""
    features: int
    name: str
    eta: float = 0.0  # utility decay, off by default
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

        self.variable("cbp", f"{self.name}_util", lambda: jnp.zeros((self.features,)))
        self.variable("cbp", f"{self.name}_age", lambda: jnp.zeros((self.features,), jnp.int32))
        self.variable("cbp", f"{self.name}_ctr", lambda: jnp.zeros(()))  # scalar float

    def __call__(self, x, next_kernel, train: bool):
        # Forward pass
        y = self.dense(x)
        h = self.act_fn(y)
        if train:
            # fetch existing variables
            util = self.get_variable("cbp", f"{self.name}_util")
            age = self.get_variable("cbp", f"{self.name}_age")

            # ------------ CBP utility + age update ------------
            w_abs_sum = jnp.sum(jnp.abs(next_kernel),
                                axis=1)  # (n_units,)  # Flax weights are saved as (in, out). PyTorch is (out, in).
            abs_neuron_output = jnp.abs(h)
            contrib = jnp.mean(abs_neuron_output,
                               axis=0) * w_abs_sum  # contribution =  |h| * Σ|w_out|  Mean over the batch (dim 0 of h)
            new_util = util * self.eta + (1 - self.eta) * contrib

            self.put_variable("cbp", f"{self.name}_util", new_util)
            self.put_variable("cbp", f"{self.name}_age", age + 1)
        return h
