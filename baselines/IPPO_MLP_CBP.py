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
            age  = self.get_variable("cbp", f"{self.name}_age")

            # ------------ CBP utility + age update ------------
            w_abs_sum = jnp.sum(jnp.abs(next_kernel), axis=1)  # (n_units,)  # Flax weights are saved as (in, out). PyTorch is (out, in).
            abs_neuron_output = jnp.abs(h)  
            contrib = jnp.mean(abs_neuron_output, axis=0) * w_abs_sum   # contribution =  |h| * Σ|w_out|  Mean over the batch (dim 0 of h)
            new_util = util * self.eta + (1 - self.eta) * contrib

            self.put_variable("cbp", f"{self.name}_util", new_util)
            self.put_variable("cbp", f"{self.name}_age",  age + 1)
        return h


# -----------------------------------------------------------
class ActorCritic(nn.Module):
    """Two-layer actor & critic with CBP-enabled hidden layers."""
    action_dim: int
    activation: str = "tanh"
    cbp_eta: float = 0.0   # CBP utility decay, off by default

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
        k_a2   = self._maybe_kernel("actor_fc2_d",  dummy_hid)
        k_aout = self._maybe_kernel("actor_out",    dummy_out)
        k_c2   = self._maybe_kernel("critic_fc2_d", dummy_hid)
        k_cout = self._maybe_kernel("critic_out",   (128, 1))

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
        next_layer_no_d = next_layer_name.split("_d")[0]  # remove "_d" from name
        util = s[layer_name][f"{layer_name}_util"]
        age = s[layer_name][f"{layer_name}_age"]
        ctr = s[layer_name][f"{layer_name}_ctr"]
        mature_mask = age > maturity
        n_eligible = jnp.sum(mature_mask).astype(int)
        ctr += n_eligible * rho

        # number of whole neurons to replace
        n_rep = jnp.floor(ctr).astype(int)
        ctr -= jnp.float32(n_rep)
        s[layer_name][f"{layer_name}_ctr"] = ctr

        util_masked = jnp.where(mature_mask, util, jnp.inf)
        sorted_idx = jnp.argsort(util_masked)  # indices ascending by util
        # build inverse permutation to get each index’s rank
        ranks = jnp.zeros_like(sorted_idx).at[sorted_idx].set(jnp.arange(util.shape[0]))
        rep_mask = (ranks < n_rep) & mature_mask  # boolean mask of size n_units

        W_in = p[layer_name][f"{layer_name}_d"]["kernel"]
        b_in = p[layer_name][f"{layer_name}_d"]["bias"]
        if '_out' in next_layer_name:  # handle last layer (actor_out, critic_out) differently
            W_out = p[next_layer_name]["kernel"]
        else:
            W_out = p[next_layer_no_d][next_layer_name]["kernel"]
        rng, k_init = jax.random.split(rng)
        new_W = weight_reinit(k_init, W_in.shape)

        # apply mask
        W_in = jnp.where(rep_mask[None, :], new_W, W_in)
        b_in = jnp.where(rep_mask, 0.0, b_in)
        W_out = jnp.where(rep_mask[:, None], 0.0, W_out)

        # reset bookkeeping
        util = jnp.where(rep_mask, 0.0, util)
        age = jnp.where(rep_mask, 0, age)

        # write back
        p[layer_name][f"{layer_name}_d"]["kernel"] = W_in
        p[layer_name][f"{layer_name}_d"]["bias"] = b_in
        if '_out' in next_layer_name:
            p[next_layer_name]["kernel"] = W_out
        else:
            p[next_layer_no_d][next_layer_name]["kernel"] = W_out
        s[layer_name][f"{layer_name}_util"] = util
        s[layer_name][f"{layer_name}_age"] = age

        return rng

    rng = _layer("actor_fc1", "actor_fc2_d", rng)
    rng = _layer("actor_fc2", "actor_out", rng)
    rng = _layer("critic_fc1", "critic_fc2_d", rng)
    rng = _layer("critic_fc2", "critic_out", rng)

    return freeze(p), freeze(s), rng


class TrainStateCBP(TrainState):
    """TrainState with an *extra* collection that stores CBP variables."""
    cbp_state: FrozenDict


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
    total_timesteps: float = 8e6
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
    alg_name: str = "ippo_cbp"
    # ------ CBP (Continual Backprop) ------
    cbp_replace_rate: float = 1e-4
    cbp_maturity: int = 10_000
    cbp_decay: float = 0.0

    seq_length: int = 6
    strategy: str = "random"
    layouts: Optional[Sequence[str]] = field(default_factory=lambda: ["asymm_advantages", "smallest_kitchen", "cramped_room", "easy_layout", "square_arena", "no_cooperation"])
    env_kwargs: Optional[Sequence[dict]] = None
    layout_name: Optional[Sequence[str]] = None
    log_interval: int = 75 # log every 200 calls to update step
    eval_num_steps: int = 1000 # number of steps to evaluate the model
    eval_num_episodes: int = 5 # number of episodes to evaluate the model
    
    anneal_lr: bool = False
    seed: int = 30
    num_seeds: int = 1
    
    # Wandb settings
    wandb_mode: str = "online"
    entity: Optional[str] = ""
    project: str = "COOX"
    tags: List[str] = field(default_factory=list)

    # to be computed during runtime
    num_actors: int = 0
    num_updates: int = 0
    minibatch_size: int = 0

    
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

def calculate_sparsity(params, threshold=1e-5):
    """
    Calculate the percentage of parameters that are close to zero
    """
    # Flatten the params into a single array
    flat_params, _ = jax.tree_util.tree_flatten(params)
    
    # Concatenate all weights into one large array
    all_weights = jnp.concatenate([jnp.ravel(p) for p in flat_params])
    
    # Count weights below threshold
    num_small_weights = jnp.sum(jnp.abs(all_weights) < threshold)
    total_weights = all_weights.size
    
    # Compute percentage of small weights
    sparsity_percentage = 100 * (num_small_weights / total_weights)
    
    return sparsity_percentage
    
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
    config.env_kwargs, config.layout_name = generate_sequence(
        sequence_length=config.seq_length,
        strategy=config.strategy,
        layout_names=config.layouts,
        seed=config.seed
    )

    for layout_config in config.env_kwargs:
        layout_name = layout_config["layout"]
        layout_config["layout"] = overcooked_layouts[layout_name]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f'{config.alg_name}_seq{config.seq_length}_{config.strategy}_{timestamp}'
    exp_dir = os.path.join("runs", run_name)

    # Initialize WandB
    load_dotenv()
    wandb_tags = config.tags if config.tags is not None else []
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=config.project,
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
            # unfreeze the environment so that we can add padding
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
    def evaluate_model(train_state, network, key):
        '''
        Evaluates the model by running 10 episodes on all environments and returns the average reward
        @param train_state: the current state of the training
        @param config: the configuration of the training
        returns the average reward
        '''

        def run_episode_while(env, key_r, network_params, max_steps=1000):
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

                def select_action(train_state, rng, obs):
                    '''
                    Selects an action based on the policy network
                    @param params: the parameters of the network
                    @param rng: random number generator
                    @param obs: the observation
                    returns the action
                    '''
                    vars_all = {"params": train_state.params, "cbp": train_state.cbp_state}
                    pi, value = train_state.apply_fn(vars_all, obs, train=False)
                    return pi.sample(seed=rng), value


                # Get action distributions
                action_a1, _ = select_action(train_state, key_a0, flat_obs["agent_0"])
                action_a2, _ = select_action(train_state, key_a1, flat_obs["agent_1"])

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

        for env in envs:
            env = make(config.env_name, layout=env)  # Create the environment
            network_params = train_state.params
            # Run k episodes
            all_rewards = jax.vmap(lambda k: run_episode_while(env, k, network_params, config.eval_num_steps))(
                jax.random.split(key, config.eval_num_episodes)
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


    network = ActorCritic(temp_env.action_space().n, activation=config.activation, cbp_eta=config.cbp_decay)

    # Initialize the network
    rng = jax.random.PRNGKey(config.seed)
    rng, network_rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space().shape).flatten()
    variables = network.init(network_rng, init_x, train=True)
    network_params = variables["params"]
    cbp_state = variables["cbp"]

    # Initialize the optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
    )

    # comment out for CBP
    # network.apply = jax.jit(network.apply)

    # Initialize the training state
    train_state = TrainStateCBP.create(
        apply_fn=network.apply,
        params=freeze(network_params),
        tx=tx,
        cbp_state=freeze(cbp_state),
    )

    def apply_net(ts: TrainStateCBP, obs, *, train_flag):
        vars_all = {"params": ts.params, "cbp": ts.cbp_state}
        (pi, val), mut = ts.apply_fn(vars_all, obs, train=train_flag, mutable=["cbp"])
        ts = ts.replace(cbp_state=freeze(mut["cbp"]))
        return ts, pi, val

    apply_net = jax.jit(apply_net, static_argnames=("train_flag",))


    # Load the practical baseline yaml file as a dictionary
    # repo_root = "/home/luka/repo/JAXOvercooked"
    repo_root = Path(__file__).resolve().parent.parent
    yaml_loc = os.path.join(repo_root, "practical_reward_baseline_results.yaml")
    with open(yaml_loc, "r") as f:
        practical_baselines = OmegaConf.load(f)

    @partial(jax.jit, static_argnums=(2))
    def train_on_environment(rng, train_state, env, env_counter):
        '''
        Trains the network using IPPO
        @param rng: random number generator
        returns the runner state and the metrics
        '''

        print("Training on environment")

        # reset the learning rate and the optimizer
        tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=linear_schedule if config.anneal_lr else config.lr, eps=1e-5)
        )
        train_state = train_state.replace(tx=tx)

        # Initialize and reset the environment
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN
        # @profile
        def _update_step(runner_state, unused):
            '''
            perform a single update step in the training loop
            @param runner_state: the carry state that contains all important training information
            returns the updated runner state and the metrics
            '''

            # COLLECT TRAJECTORIES
            # @profile
            def _env_step(runner_state, unused):
                '''
                selects an action based on the policy, calculates the log probability of the action,
                and performs the selected action in the environment
                @param runner_state: the current state of the runner
                returns the updated runner state and the transition
                '''
                # Unpack the runner state
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                # split the random number generator for action selection
                rng, _rng = jax.random.split(rng)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, env.agents, config.num_actors)
                # print("obs_shape", obs_batch.shape)

                # fwd-pass the network on the observations to get the suggested actions and their values
                # pi, value = network.apply(train_state.params, obs_batch)
                train_state, pi, value = apply_net(train_state, obs_batch, train_flag=True)

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

                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, (transition, info)

            # Run the _env_step function a series of times, while keeping track of the runner state
            runner_state, (traj_batch, info) = jax.lax.scan(
                f=_env_step,
                init=runner_state,
                xs=None,
                length=config.num_steps
            )

            # unpack the runner state that is returned after the scan function
            train_state, env_state, last_obs, update_step, rng = runner_state

            # create a batch of the observations that is compatible with the network
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors)

            # fwd-pass the network on the batch of observations to get the value of the last state
            # _, last_val = network.apply(train_state.params, last_obs_batch)
            vars_all = {"params": train_state.params, "cbp": train_state.cbp_state}
            _, last_val = train_state.apply_fn(vars_all, last_obs_batch, train=False)
            # this returns the value network for the last observation batch

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

                # iteratively run the _get_advantages function to calculate the advantage for each step in the trajectory batch
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

                def _update_minbatch(carry, batch_info):
                    '''
                    performs a single update minibatch in the training loop
                    @param carry: (the current state of the training, rng)
                    @param batch_info: the information of the batch
                    returns the updated train_state and the total loss
                    '''
                    # unpack the information
                    train_state, rng = carry
                    traj_batch, advantages, targets = batch_info

                    # @profile
                    def _loss_fn(params, traj_batch, gae, targets):
                        '''
                        calculates the loss of the network
                        @param params: the parameters of the network
                        @param traj_batch: the trajectory batch
                        @param gae: the generalized advantage estimate
                        @param targets: the targets
                        @param network: the network
                        returns the total loss and the value loss, actor loss, and entropy
                        '''
                        # fwd-pass the network on the observations in the trajectory batch
                        # pi, value = network.apply(params, traj_batch.obs)
                        vars_all = {"params": params, "cbp": train_state.cbp_state}
                        pi, value = train_state.apply_fn(vars_all, traj_batch.obs, train=False)
                        log_prob = pi.log_prob(traj_batch.action)

                        # calculate critic loss
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

                        # Calculate actor loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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

                        loss_actor = -jnp.minimum(loss_actor_unclipped,
                                                  loss_actor_clipped)  # calculate the actor loss as the minimum of the clipped and unclipped actor loss
                        loss_actor = loss_actor.mean()  # calculate the mean of the actor loss
                        entropy = pi.entropy().mean()  # calculate the entropy of the policy

                        total_loss = (
                                loss_actor
                                + config.vf_coef * value_loss
                                - config.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    # returns a function with the same parameters as loss_fn that calculates the gradient of the loss function
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    # call the grad_fn function to get the total loss and the gradients
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                    loss_information = total_loss, grads

                    # add the gradients to the network parameters
                    train_state = train_state.apply_gradients(grads=grads)

                    # ---- Continual-Backprop neuron replacement ----
                    rng, cbp_rng = jax.random.split(rng)
                    new_p, new_c, _ = cbp_step(train_state.params,
                                               train_state.cbp_state,
                                               rng=cbp_rng,
                                               maturity=config.cbp_maturity,
                                               rho=config.cbp_replace_rate)
                    train_state = train_state.replace(params=new_p, cbp_state=new_c)

                    # Of course we also need to add the network to the carry here
                    return (train_state, rng), loss_information

                # unpack the update_state (because of the scan function)
                train_state, traj_batch, advantages, targets, rng = update_state

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
                )  # outputs a tuple of the batch, advantages, and targets shuffled

                minibatches = jax.tree_util.tree_map(
                f=(lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:]))), tree=shuffled_batch,
                )

                carry_init = (train_state, rng)
                (train_state, rng), loss_information = jax.lax.scan(
                f=_update_minbatch,
                init=carry_init,
                xs=minibatches
                )

                total_loss, grads = loss_information
                avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # create a tuple to be passed into the jax.lax.scan function
            update_state = (train_state, traj_batch, advantages, targets, rng)

            update_state, loss_info = jax.lax.scan(
                f=_update_epoch,
                init=update_state,
                xs=None,
                length=config.update_epochs
            )

            # unpack update_state
            train_state, traj_batch, advantages, targets, rng = update_state

            # set the metric to be the information of the last update epoch
            metric = info

            # calculate the current timestep
            current_timestep = update_step * config.num_steps * config.num_envs

            # average the metric
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

            update_step += 1

            # add the general metrics to the metric dictionary
            metric["General/update_step"] = update_step
            metric["General/env_step"] = update_step * config.num_steps * config.num_envs
            metric["General/learning_rate"] = linear_schedule(update_step * config.num_minibatches * config.update_epochs)
            metric["General/sparsity"] = calculate_sparsity(train_state.params)

            # Losses section
            total_loss, (value_loss, loss_actor, entropy) = loss_info
            metric["Losses/total_loss"] = total_loss.mean()
            metric["Losses/value_loss"] = value_loss.mean()
            metric["Losses/actor_loss"] = loss_actor.mean()
            metric["Losses/entropy"] = entropy.mean()

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
                metric[f"Scaled returns/evaluation_{config.layout_name[i]}_scaled"] = jnp.nan

            def evaluate_and_log(rng, update_step):
                rng, eval_rng = jax.random.split(rng)
                train_state_eval = jax.tree_util.tree_map(lambda x: x.copy(), train_state)

                def log_metrics(metric, update_step):
                    evaluations = evaluate_model(train_state_eval, network, eval_rng)
                    for i, layout_name in enumerate(config.layout_name):
                        metric[f"Evaluation/{layout_name}"] = evaluations[i]

                        # Add error handling for missing baseline entries
                        bare_layout = layout_name.split("__")[1]
                        try:
                            if bare_layout in practical_baselines:
                                baseline_reward = practical_baselines[bare_layout]["avg_rewards"]
                                metric[f"Scaled returns/evaluation_{layout_name}_scaled"] = evaluations[i] / baseline_reward
                            else:
                                print(f"Warning: No baseline data for environment '{bare_layout}'")
                                # Use 1.0 as default normalization factor (no scaling)
                                metric[f"Scaled returns/evaluation_{layout_name}_scaled"] = evaluations[i]
                        except Exception as e:
                            print(f"Error scaling rewards for {layout_name}: {e}")
                            metric[f"Scaled returns/evaluation_{layout_name}_scaled"] = evaluations[i]

                    params = jax.tree_util.tree_map(lambda x: x, train_state_eval.params)  # ["params"])

                    # Same error handling for callback function
                    def callback(args):
                        metric, update_step, env_counter = args
                        real_step = (int(env_counter) - 1) * config.num_updates + int(update_step)

                        env_name = config.layout_name[env_counter - 1]
                        if env_name in practical_baselines:
                            metric["Scaled returns/returned_episode_returns_scaled"] = metric["returned_episode_returns"] / practical_baselines[env_name]["avg_rewards"]
                        else:
                            metric["Scaled returns/returned_episode_returns_scaled"] = metric["returned_episode_returns"]  # No scaling if no baseline

                        for key, value in metric.items():
                            writer.add_scalar(key, value, real_step)

                        # Add histograms of the parameters and grads to the writer
                        for layer, dict in params.items():
                            for layer_name, param_array in dict.items():
                                writer.add_histogram(
                                    tag=f"weights/{layer}/{layer_name}",
                                    values=jnp.array(param_array),
                                    global_step=real_step,
                                    bins=100)

                    jax.experimental.io_callback(callback, None, (metric, update_step, env_counter))
                    return None

                def do_not_log(metric, update_step):
                    return None

                jax.lax.cond((update_step % config.log_interval) == 0, log_metrics, do_not_log, metric, update_step)

            # Evaluate the model and log the metrics
            evaluate_and_log(rng=rng, update_step=update_step)

            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, update_step, rng)

            return runner_state, metric

        rng, train_rng = jax.random.split(rng)

        # initialize a carrier that keeps track of the states and observations of the agents
        runner_state = (train_state, env_state, obsv, 0, train_rng)

        # run the _update_step function a series of times, while keeping track of the state
        runner_state, metric = jax.lax.scan(
            f=_update_step,
            init=runner_state,
            xs=None,
            length=config.num_updates
        )

        # Return the runner state after the training loop, and the metric arrays
        return runner_state, metric

    def loop_over_envs(rng, train_state, envs):
        '''
        Loops over the environments and trains the network
        @param rng: random number generator
        @param train_state: the current state of the training
        @param envs: the environments
        returns the runner state and the metrics
        '''
        # split the random number generator for training on the environments
        rng, *env_rngs = jax.random.split(rng, len(envs) + 1)

        # counter for the environment
        env_counter = 1

        for env_rng, env in zip(env_rngs, envs):
            runner_state, metrics = train_on_environment(env_rng, train_state, env, env_counter)

            # unpack the runner state
            train_state, env_state, last_obs, update_step, rng = runner_state

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

    # Run the model
    rng, train_rng = jax.random.split(rng)
    # run the loop_over_envs function to the environments
    runner_state = loop_over_envs(train_rng, train_state, envs)


def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)


if __name__ == "__main__":
    print("Running main...")
    main()

