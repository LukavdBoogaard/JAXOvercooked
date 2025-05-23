from collections import OrderedDict
from enum import IntEnum

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax_marl.gridworld.env import Environment
from jax_marl.environments import spaces
from typing import Tuple
import chex
from flax import struct

from jax_marl.environments.overcooked_environment.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    DIR_TO_VEC,
    make_maze_map)


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5

    # Done completing task
    done = 6


@struct.dataclass
class EnvState:
    agent_pos: chex.Array
    agent_dir: chex.Array
    agent_dir_idx: int
    goal_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool


@struct.dataclass
class EnvParams:
    height: int = 15
    width: int = 15
    n_walls: int = 25
    agent_view_size: int = 5
    replace_wall_pos: bool = False
    see_through_walls: bool = True
    see_agent: bool = False
    normalize_obs: bool = False
    sample_n_walls: bool = False  # Sample n_walls uniformly in [0, n_walls]
    max_episode_steps: int = 250
    singleton_seed: int = -1
    n_agents: int = 2


class MAMaze(Environment):
    """Multi-agent Maze"""
    def __init__(
            self,
            height=13,
            width=13,
            n_walls=25,
            agent_view_size=5,
            replace_wall_pos=False,
            see_agent=False,
            max_episode_steps=250,
            normalize_obs=False,
            singleton_seed=-1,
            n_agents=2
    ):
        super().__init__()

        self.obs_shape = (agent_view_size, agent_view_size, 3)
        self.action_set = jnp.array([
            Actions.left,
            Actions.right,
            Actions.forward,
            Actions.pickup,
            Actions.drop,
            Actions.toggle,
            Actions.done
        ])

        self.params = EnvParams(
            height=height,
            width=width,
            n_walls=n_walls,
            agent_view_size=agent_view_size,
            replace_wall_pos=replace_wall_pos and not sample_n_walls,
            see_agent=see_agent,
            max_episode_steps=max_episode_steps,
            normalize_obs=normalize_obs,
            singleton_seed=-1,
            n_agents=n_agents
        )

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            actions: chex.Array,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""

        acts = self.action_set.take(indices=jnp.array(actions))

        state, reward = self.step_agents(key, state, acts)
        # Check game condition & no. steps for termination condition
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs_array = jax.vmap(self.get_obs, in_axes=(None, 0))(state, jnp.arange(self.params.n_agents))

        return (
            lax.stop_gradient(obs_array),
            lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            {},
        )

    def reset_env(
            self,
            key: chex.PRNGKey,
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by resampling contents of maze_map
        - initial agent position
        - goal position
        - wall positions
        """
        params = self.params
        h = params.height
        w = params.width
        n_agents = params.n_agents
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        # Reset wall map, with shape H x W, and value of 1 at (i,j) iff there is a wall at (i,j)
        key, subkey = jax.random.split(key)
        wall_idx = jax.random.choice(
            subkey, all_pos,
            shape=(params.n_walls,),
            replace=params.replace_wall_pos)

        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

        # Reset agent position + dir
        key, subkey = jax.random.split(key)
        agent_idx = jax.random.choice(subkey, all_pos, shape=(n_agents,),
                                      p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32), replace=False)
        occupied_mask = occupied_mask.at[agent_idx].set(1)
        agent_pos = jnp.array([agent_idx % w, agent_idx // w], dtype=jnp.uint32).transpose() # dim = n_agents x 2

        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.uint8), shape=(n_agents,))
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get() # dim = n_agents x 2

        # Reset goal position
        key, subkey = jax.random.split(key)
        goal_idx = jax.random.choice(subkey, all_pos, shape=(1,),
                                     p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32))
        goal_pos = jnp.array([goal_idx % w, goal_idx // w], dtype=jnp.uint32).flatten()

        maze_map = make_maze_map(
            params,
            wall_map,
            goal_pos,
            agent_pos,
            agent_dir_idx,
            pad_obs=True)

        state = EnvState(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            goal_pos=goal_pos,
            wall_map=wall_map.astype(jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
        )
        obs_array = jax.vmap(self.get_obs, in_axes=(None, 0))(state, jnp.arange(n_agents))

        return obs_array, state

    def get_obs(self, state: EnvState, agent_idx: int) -> chex.Array:
        """Return limited grid view ahead of agent."""
        obs = jnp.zeros(self.obs_shape, dtype=jnp.uint8)
        agent_pos = state.agent_pos[agent_idx]
        agent_dir = state.agent_dir[agent_idx]
        agent_dir_idx = state.agent_dir_idx[agent_idx]

        agent_x, agent_y = agent_pos

        obs_fwd_bound1 = agent_pos
        obs_fwd_bound2 = agent_pos + agent_dir * (self.obs_shape[0] - 1)

        side_offset = self.obs_shape[0] // 2
        obs_side_bound1 = agent_pos + (agent_dir == 0) * side_offset
        obs_side_bound2 = agent_pos - (agent_dir == 0) * side_offset

        all_bounds = jnp.stack([obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2])

        # Clip obs to grid bounds appropriately
        padding = obs.shape[0] - 1
        obs_bounds_min = np.min(all_bounds, 0) + padding
        obs_range_x = jnp.arange(obs.shape[0]) + obs_bounds_min[1]
        obs_range_y = jnp.arange(obs.shape[0]) + obs_bounds_min[0]

        meshgrid = jnp.meshgrid(obs_range_y, obs_range_x)
        coord_y = meshgrid[1].flatten()
        coord_x = meshgrid[0].flatten()

        obs = state.maze_map.at[
              coord_y, coord_x, :].get().reshape(obs.shape[0], obs.shape[1], 3)

        obs = (agent_dir_idx == 0) * jnp.rot90(obs, 1) + \
              (agent_dir_idx == 1) * jnp.rot90(obs, 2) + \
              (agent_dir_idx == 2) * jnp.rot90(obs, 3) + \
              (agent_dir_idx == 3) * jnp.rot90(obs, 4)

        if not self.params.see_agent:
            obs = obs.at[-1, side_offset].set(
                jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
            )

        image = obs.astype(jnp.uint8)
        if self.params.normalize_obs:
            image = image / 10.0

        obs_dict = dict(
            image=image,
            agent_dir=agent_dir_idx
        )

        return OrderedDict(obs_dict)

    def step_agents(self, key: chex.PRNGKey, state: EnvState, action: chex.Array) -> Tuple[EnvState, float]:
        params = self.params

        # Update agent position (forward action)
        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + (action == Actions.forward) * state.agent_dir, 0),
            jnp.array((params.width - 1, params.height - 1), dtype=jnp.uint32))

        # Can't go past wall or goal
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            fwd_goal = jnp.logical_and(fwd_position[0] == goal_pos[0], fwd_position[1] == goal_pos[1])
            return fwd_wall, fwd_goal

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(_wall_or_goal, in_axes=(0, None, None))(fwd_pos, state.wall_map, state.goal_pos)

        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal).reshape((params.n_agents, 1))

        # Agents can't overlap
        # Hardcoded for 2 agents
        agent_pos_prev = jnp.array(state.agent_pos)
        fwd_pos = (fwd_pos_blocked * state.agent_pos + (~fwd_pos_blocked) * fwd_pos).astype(jnp.uint32)
        collision = jnp.all(fwd_pos[0] == fwd_pos[1])

        # Cases (bounced == hit wall/goal and stayed in place)
        neither_bounced = (~fwd_pos_blocked).all()
        only_alice_bounced = jnp.logical_and(fwd_pos_blocked[0], ~fwd_pos_blocked[1])
        only_bob_bounced = jnp.logical_and(~fwd_pos_blocked[0], fwd_pos_blocked[1])

        alice_pos = jnp.where(
            collision * only_bob_bounced,
            state.agent_pos[0],                     # collision and Bob bounced
            fwd_pos[0],
        )
        bob_pos = jnp.where(
            collision * only_alice_bounced,
            state.agent_pos[1],                     # collision and Alice bounced
            fwd_pos[1],
        )

        key, rng_coll = jax.random.split(key)
        alice_wins = jax.random.choice(rng_coll, 2)

        alice_pos = jnp.where(
            collision * neither_bounced * (1-alice_wins),
            state.agent_pos[0],
            alice_pos
        )
        bob_pos = jnp.where(
            collision * neither_bounced * alice_wins,
            state.agent_pos[1],
            bob_pos
        )

        # Prevent swapping places (i.e. passing through each other)
        swap_places = jnp.logical_and(
            jnp.all(fwd_pos[0] == state.agent_pos[1]),
            jnp.all(fwd_pos[1] == state.agent_pos[0]),
        )
        alice_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[0],
            alice_pos
        )
        bob_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[1],
            bob_pos
        )

        fwd_pos = fwd_pos.at[0].set(alice_pos)
        fwd_pos = fwd_pos.at[1].set(bob_pos)
        agent_pos = fwd_pos.astype(jnp.uint32)

        # Update agent direction (left_turn or right_turn action)
        agent_dir_offset = \
            0 \
            + (action == Actions.left) * (-1) \
            + (action == Actions.right) * 1

        agent_dir_idx = (state.agent_dir_idx + agent_dir_offset) % 4
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # # Update agent component in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev, agent_idx):
            agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red']+agent_idx*2, agent_dir_idx], dtype=jnp.uint8)
            agent_x_prev, agent_y_prev = agent_pos_prev
            agent_x, agent_y = agent_pos
            return agent_x, agent_y, agent_x_prev, agent_y_prev, agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(agent_dir_idx, agent_pos, agent_pos_prev, jnp.arange(params.n_agents))
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
        padding = self.obs_shape[0] - 1

        maze_map = state.maze_map
        maze_map = maze_map.at[padding + agent_y_prev, padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y, padding + agent_x, :].set(agent_vec)

        reward = jnp.max((1.0 - 0.9 * ((state.time + 1) / params.max_episode_steps)) * fwd_pos_has_goal)

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                maze_map=maze_map,
                terminal=fwd_pos_has_goal.any()),
            reward
        )

    def is_terminal(self, state: EnvState) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.params.max_episode_steps
        return jnp.logical_or(done_steps, state.terminal)

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats['return'] > 0

        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return "Maze"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(
            len(self.action_set),
            dtype=jnp.uint32
        )

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        spaces_dict = {
            'image': spaces.Box(0, 255, self.obs_shape),
            'agent_dir': spaces.Discrete(4)
        }

        return spaces.Dict(spaces_dict)

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        params = self.params
        h = params.height
        w = params.width
        agent_view_size = params.agent_view_size
        return spaces.Dict({
            "agent_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "agent_dir": spaces.Discrete(4),
            "goal_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "maze_map": spaces.Box(0, 255, (w + agent_view_size, h + agent_view_size, 3), dtype=jnp.uint32),
            "time": spaces.Discrete(params.max_episode_steps),
            "terminal": spaces.Discrete(2),
        })

    def max_episode_steps(self) -> int:
        return self.params.max_episode_steps