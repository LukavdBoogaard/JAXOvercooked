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
                    network_apply = train_state.apply_fn
                    pi, value = network_apply(network_params, obs)
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

    # jit the apply function
    network.apply = jax.jit(network.apply)

    # Initialize the training state
    train_state = TrainStateCBP.create(
        apply_fn=network.apply,
        params=freeze(network_params),
        tx=tx,
        cbp_state=freeze(cbp_state),
    )

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
                                        jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * adv))
        v_clip = batch.value + (value - batch.value).clip(-config.clip_eps, config.clip_eps)
        v_loss = 0.5 * jnp.mean(jnp.maximum((value - batch.ret) ** 2,
                                            (v_clip - batch.ret) ** 2))
        ent = jnp.mean(pi.entropy())
        loss = loss_pi + config.vf_coef * v_loss - config.ent_coef * ent
        return loss, (ts2, loss_pi, v_loss, ent)

    # jit for speed (purely functional)
    loss_grad = jax.jit(jax.value_and_grad(loss_grad, has_aux=True), static_argnums=2)

    # ---------------- curriculum loop -----------------
    global_step = 0
    for env_idx, env_cfg in enumerate(config.env_kwargs, start=1):
        env = make(config.env_name, **env_cfg)
        env = LogWrapper(env, replace_info=False)

        config.num_actors = env.num_agents * config.num_envs
        config.num_updates = int(config.total_timesteps // (config.num_envs * config.num_steps))
        config.minibatch_size = config.num_actors * config.num_steps // config.num_minibatches

        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config.num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_rng)

        # ------------- per-environment training loop -------------
        for upd in range(config.num_updates):
            traj = []

            # ---------- COLLECT ----------
            for _ in range(config.num_steps):
                rng, step_rng = jax.random.split(rng)
                ts, pi, v = apply_net(train_state, batchify(obs, env.agents,
                                                            config.num_actors),
                                      train_flag=True, rng=step_rng)
                act = pi.sample(seed=step_rng)
                logp = pi.log_prob(act)
                env_act = unbatchify(act, env.agents, config.num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}
                rng, step_rng = jax.random.split(rng)
                step_rngs = jax.random.split(step_rng, config.num_envs)
                obs2, env_state, rew, done, info = jax.vmap(env.step)(step_rngs,
                                                                      env_state,
                                                                      env_act)
                traj.append(Transition(batchify(done, env.agents, config.num_actors).squeeze(),
                                       act, v, batchify(rew, env.agents,
                                                        config.num_actors).squeeze(),
                                       logp, batchify(obs, env.agents,
                                                      config.num_actors)))
                obs = obs2

                train_state = ts  # cbp_state already updated

            # --------- ADV / RETURN (GAE) ----------
            last_obs_batch = batchify(obs, env.agents, config.num_actors)
            _, last_val = train_state.apply_fn(
                {"params": train_state.params, **train_state.cbp_state},
                last_obs_batch, train=False)
            traj = Transition(*map(jnp.stack, zip(*traj)))  # stack along time

            def _gae(carry, t):
                gae, next_val = carry
                d, v, r = t.done, t.value, t.reward
                delta = r + config.gamma * next_val * (1 - d) - v
                gae = delta + config.gamma * config.gae_lambda * (1 - d) * gae
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
            for epoch in range(config.update_epochs):
                perm = jax.random.permutation(rng,
                                              train_state.params["actor_out"]["bias"].shape[0] * config.num_steps)
                perm = perm.reshape((config.num_minibatches, -1))
                for idx in perm:
                    mini = jax.tree_util.tree_map(lambda x: x[idx], flat)
                    rng, grad_rng = jax.random.split(rng)
                    (aux, grads), grads_val = loss_grad(train_state, mini, grad_rng)
                    ts2, l_pi, l_v, ent = aux
                    train_state = ts2.apply_gradients(grads=grads_val)

                    # ------ CBP maintenance (after optimizer step) ------
                    rng, cbp_rng = jax.random.split(rng)
                    new_params, new_cstate, cbp_rng = cbp_step(
                        train_state.params,
                        train_state.cbp_state,
                        rng=cbp_rng,
                        maturity=config.cbp_maturity,
                        rho=config.cbp_replace_rate)
                    train_state = train_state.replace(params=new_params, cbp_state=new_cstate)

            global_step += config.num_envs * config.num_steps

            if upd % config.log_interval == 0:
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


def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)


if __name__ == "__main__":
    print("Running main...")
    main()

