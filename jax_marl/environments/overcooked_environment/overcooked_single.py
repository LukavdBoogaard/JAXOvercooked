#!/usr/bin/env python
"""Single‑agent JAX Overcooked environment.

This is a stripped‑down fork of the original 2‑agent environment that ships with
`jax‑marl`.  All multi‑agent collision logic and opponent‑specific observation
layers have been binned, so you can drop it into a Gym‑style pipeline or a JAX
RL training loop with zero hassle.  Nothing fancy here: one chef, one brain.

"""

from enum import IntEnum
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from jax_marl.environments import spaces, MultiAgentEnv
from jax_marl.environments.overcooked_environment.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map,
)
from jax_marl.environments.overcooked_environment.layouts import (
    overcooked_layouts as layouts,
)

# -----------------------------------------------------------------------------
# Shared constants grabbed from original code (no need to import mx of modules)
# -----------------------------------------------------------------------------

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "PLATE_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DROP_COUNTER_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

# Pot state machine
POT_EMPTY_STATUS = 23
POT_FULL_STATUS = 20
POT_READY_STATUS = 0
MAX_ONIONS_IN_POT = 3
URGENCY_CUTOFF = 40
DELIVERY_REWARD = jnp.float32(20.0)


class Actions(IntEnum):
    up = 0
    down = 1
    right = 2
    left = 3
    stay = 4
    interact = 5
    done = 6  # noop: maintained for API parity with jax‑marl


@struct.dataclass
class State:
    agent_pos: chex.Array  # (2,)  x,y
    agent_dir: chex.Array  # (2,)
    agent_dir_idx: chex.Array  # ()
    agent_inv: chex.Array  # ()  holds OBJECT_TO_INDEX value
    goal_pos: chex.Array  # (n_goals,2)
    pot_pos: chex.Array  # (n_pots,2)
    wall_map: chex.Array  # (H,W) bool
    maze_map: chex.Array  # (H+pad,W+pad,3) uint8
    time: int
    terminal: bool


class OvercookedSingle(MultiAgentEnv):
    """One‑chef Overcooked."""

    def __init__(
            self,
            *,
            layout: Dict | None = None,
            layout_name: str = "cramped_room",
            random_reset: bool = False,
            max_steps: int = 400,
    ) -> None:
        super().__init__(num_agents=1)
        self.layout = layout if layout is not None else layouts[layout_name]
        self.layout_name = layout_name
        self.random_reset = random_reset
        self.max_steps = max_steps

        self.width = self.layout["width"]
        self.height = self.layout["height"]
        self.agent_view_size = 5  # padding used by make_overcooked_map
        self.obs_shape = (self.width, self.height, 26)  # keep original depth

        self.action_set = jnp.array(
            [
                Actions.up,
                Actions.down,
                Actions.right,
                Actions.left,
                Actions.stay,
                Actions.interact,
            ],
            dtype=jnp.int32,
        )

        # Gym‑style agent ID list for compatibility
        self.agents = ["agent_0"]

    # ------------------------------------------------------------------
    # API: reset / step
    # ------------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Randomised or deterministic reset."""
        h, w = self.height, self.width
        all_pos = jnp.arange(h * w, dtype=jnp.uint32)
        wall_idx = jnp.asarray(self.layout["wall_idx"], dtype=jnp.uint32)

        wall_map = jnp.zeros_like(all_pos, dtype=jnp.bool_)
        wall_map = wall_map.at[wall_idx].set(True)
        wall_map = wall_map.reshape(h, w)

        # Spawn position
        key, subkey = jax.random.split(key)
        spawn_idx = None
        if not self.random_reset:
            spawn_locs = self.layout.get("agent_idx", None)
            if spawn_locs is not None:
                spawn_idx = jnp.asarray(spawn_locs)[0]

        if spawn_idx is None:
            valid = ~wall_map.flatten()
            spawn_idx = jax.random.choice(subkey, all_pos, p=valid.astype(jnp.float32))

        agent_pos = jnp.array([spawn_idx % w, spawn_idx // w], dtype=jnp.uint32)

        # Orientation
        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.randint(subkey, (), 0, 4)
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Inventory (always empty on deterministic resets)
        agent_inv = (
            OBJECT_TO_INDEX["empty"]
            if not self.random_reset
            else int(jax.random.choice(subkey, jnp.array([OBJECT_TO_INDEX[x] for x in ["empty", "onion", "plate", "dish"]])))
        )

        goal_pos = self._fixed_vec("goal_idx")
        pot_pos = self._fixed_vec("pot_idx")

        # Build initial maze map with helper from original impl.
        maze_map = make_overcooked_map(
            wall_map,
            goal_pos,
            agent_pos[None, :],  # expects N x 2
            jnp.array([agent_dir_idx]),
            self._fixed_vec("plate_pile_idx"),
            self._fixed_vec("onion_pile_idx"),
            pot_pos,
            jnp.ones(pot_pos.shape[0], dtype=jnp.uint8) * POT_EMPTY_STATUS,
            onion_pos=jnp.zeros((0, 2), dtype=jnp.uint32),
            plate_pos=jnp.zeros((0, 2), dtype=jnp.uint32),
            dish_pos=jnp.zeros((0, 2), dtype=jnp.uint32),
            pad_obs=True,
            num_agents=1,
            agent_view_size=self.agent_view_size,
            )

        state = State(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=jnp.asarray(agent_inv, dtype=jnp.uint8),
            goal_pos=goal_pos,
            pot_pos=pot_pos,
            wall_map=wall_map,
            maze_map=maze_map,
            time=0,
            terminal=False,
        )
        return self.get_obs(state), lax.stop_gradient(state)

    def step_env(
            self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        act = self.action_set[actions["agent_0"]]
        state, reward, shaped = self._step_agent(state, act)
        state = state.replace(time=state.time + 1, terminal=self.is_terminal(state))
        obs = self.get_obs(state)
        done = state.terminal
        return obs, state, {"agent_0": reward}, {"agent_0": done, "__all__": done}, {"shaped_reward": {"agent_0": shaped}}

    # ------------------------------------------------------------------
    # Inner mechanics (movement + interact)  — single‑agent is simpler!
    # ------------------------------------------------------------------

    def _step_agent(self, state: State, action: chex.Array):
        """Move + interact for exactly one chef."""
        # ------------------------------------------------------------------
        # Move / turn
        # ------------------------------------------------------------------
        is_move = (action < Actions.stay).astype(jnp.uint8)
        new_dir_idx = jnp.where(is_move, action, state.agent_dir_idx)
        new_dir = DIR_TO_VEC[new_dir_idx]

        fwd_pos = jnp.clip(state.agent_pos + new_dir * is_move, 0, jnp.array([self.width - 1, self.height - 1]))
        diag_block = (
                state.wall_map[state.agent_pos[1], fwd_pos[0]] &   # wall beside
                state.wall_map[fwd_pos[1], state.agent_pos[0]]     # wall above/below
        )
        # Block on walls/goals
        blocked = state.wall_map[fwd_pos[1], fwd_pos[0]] | diag_block \
                  | jnp.any(jnp.all(fwd_pos == state.goal_pos, axis=1))
        new_pos = jnp.where(blocked, state.agent_pos, fwd_pos)
        new_pos = new_pos.astype(state.agent_pos.dtype)

        # ------------------------------------------------------------------
        # Interact
        # ------------------------------------------------------------------
        maze_map = state.maze_map

        def do_interact(args):
            m, inv, r, s = self._process_interact(maze_map, state.wall_map, new_pos, state.agent_inv)
            return (m, inv, r, s)
        def no_interact(_):
            return (maze_map, state.agent_inv, jnp.float32(0.0), jnp.float32(0.0))

        maze_map, new_inv, reward, shaped = lax.cond(action==Actions.interact, do_interact, no_interact, operand=None)

        # Update agent vec in map (remove old, add new)
        padding = (maze_map.shape[0] - self.height) // 2
        empty_vec = jnp.array([OBJECT_TO_INDEX["empty"], 0, 0], dtype=jnp.uint8)
        old_y, old_x = state.agent_pos[1], state.agent_pos[0]
        new_y, new_x = new_pos[1], new_pos[0]
        maze_map = maze_map.at[padding + old_y, padding + old_x, :].set(empty_vec)
        agent_vec = jnp.array([OBJECT_TO_INDEX["agent"], COLOR_TO_INDEX["red"], new_dir_idx], dtype=jnp.uint8)
        maze_map = maze_map.at[padding + new_y, padding + new_x, :].set(agent_vec)

        new_state = state.replace(
            agent_pos=new_pos,
            agent_dir=new_dir,
            agent_dir_idx=new_dir_idx,
            agent_inv=new_inv,
            maze_map=maze_map,
        )
        return new_state, reward, shaped

    # ------------------------------------------------------------------
    # Interaction logic — copied with minimal edits
    # ------------------------------------------------------------------

    def _process_interact(self, maze_map, wall_map, fwd_pos, inventory):
        pad = (maze_map.shape[0] - self.height) // 2
        row = jnp.ravel(fwd_pos[1])[0].astype(jnp.int32)
        col = jnp.ravel(fwd_pos[0])[0].astype(jnp.int32)
        pad_i = jnp.int32(pad)
        cell = lax.dynamic_slice(maze_map, (pad_i + row, pad_i + col, jnp.int32(0)),
                                 (1, 1, 3)).reshape((3,))
        maze_map = lax.dynamic_update_slice(maze_map,
                                            cell.reshape((1, 1, 3)),
                                            (pad + row, pad + col, 0))
        obj_id = cell[0]

        # flags ─────────────────────────────────────────────────────────────
        is_pile      = (obj_id == OBJECT_TO_INDEX["plate_pile"]) | (obj_id == OBJECT_TO_INDEX["onion_pile"])
        is_pot       =  obj_id == OBJECT_TO_INDEX["pot"]
        is_goal      =  obj_id == OBJECT_TO_INDEX["goal"]
        is_pickable  = (obj_id == OBJECT_TO_INDEX["plate"]) | (obj_id == OBJECT_TO_INDEX["onion"]) | (obj_id == OBJECT_TO_INDEX["dish"])
        table_empty  = (obj_id == OBJECT_TO_INDEX["empty"]) | (obj_id == OBJECT_TO_INDEX["wall"])
        # table_wall = wall_map[row, col]
        table_wall = jnp.all(wall_map[row, col])
        is_table   = jnp.logical_and(table_wall, ~is_pot)

        inv_empty    = inventory == OBJECT_TO_INDEX["empty"]
        holding_onion  = inventory == OBJECT_TO_INDEX["onion"]
        holding_plate  = inventory == OBJECT_TO_INDEX["plate"]
        holding_dish   = inventory == OBJECT_TO_INDEX["dish"]

        pot_status = cell[-1]

        # pot logic ─────────────────────────────────────────────────────────
        add_onion   = is_pot & (pot_status > POT_FULL_STATUS)  & holding_onion
        take_soup   = is_pot & (pot_status == POT_READY_STATUS) & holding_plate
        wait_pot    = is_pot & (POT_READY_STATUS < pot_status) & (pot_status <= POT_FULL_STATUS)

        pot_status_new = (
            jnp.where(add_onion, pot_status - 1,
                      jnp.where(take_soup, POT_EMPTY_STATUS, pot_status))
        )

        inv_new = jnp.where(add_onion | take_soup,     # emptied or got dish
                            jnp.where(take_soup, OBJECT_TO_INDEX["dish"], OBJECT_TO_INDEX["empty"]),
                            inventory)

        shaped = (BASE_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"] * add_onion +
                  BASE_REW_SHAPING_PARAMS["SOUP_PICKUP_REWARD"]   * take_soup)
        shaped = jnp.float32(shaped)

        # generic pickup / drop / delivery ──────────────────────────────────
        pickup = is_table & (~table_empty) & inv_empty & (is_pile | is_pickable)
        drop   = is_table & table_empty   & (~inv_empty)
        deliver= is_goal  & holding_dish

        # inventory transitions
        inv_new = jnp.where(pickup,
                            jnp.where(obj_id == OBJECT_TO_INDEX["plate_pile"],  OBJECT_TO_INDEX["plate"],
                                      jnp.where(obj_id == OBJECT_TO_INDEX["onion_pile"],  OBJECT_TO_INDEX["onion"],
                                                obj_id)),
                            inv_new)

        inv_new = jnp.where(drop | deliver, OBJECT_TO_INDEX["empty"], inv_new)

        # table cell update
        new_cell_id = jnp.where(drop, inventory,
                                jnp.where(pickup & is_pickable, OBJECT_TO_INDEX["wall"], obj_id))
        cell = jnp.where(is_pot,
                         OBJECT_INDEX_TO_VEC[new_cell_id].at[-1].set(pot_status_new),
                         OBJECT_INDEX_TO_VEC[new_cell_id])
        maze_map = maze_map.at[pad + fwd_pos[1], pad + fwd_pos[0], :].set(cell)

        reward = jnp.where(deliver, DELIVERY_REWARD, jnp.float32(0.0))
        return maze_map, jnp.asarray(inv_new, jnp.uint8), reward, shaped


    # ------------------------------------------------------------------
    # Obs: keep 26‑channel tensor so downstream code needs no edits
    # ------------------------------------------------------------------

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        width, height = self.width, self.height
        padding = (state.maze_map.shape[0] - height) // 2

        maze = state.maze_map[padding:-padding, padding:-padding, 0]
        # layer 0: self pos, layer 1: *dead* (zeros) to match old shape
        pos_layers = jnp.zeros((2, height, width), dtype=jnp.uint8)
        pos_layers = pos_layers.at[0, state.agent_pos[1], state.agent_pos[0]].set(1)

        # dir layers 0‑3 (self) + 4‑7 (zeros)
        dir_layers = jnp.zeros((8, height, width), dtype=jnp.uint8)
        dir_layers = dir_layers.at[state.agent_dir_idx, state.agent_pos[1], state.agent_pos[0]].set(1)

        # env layers copied wholesale from original impl. (unchanged)
        pot_layer = maze == OBJECT_TO_INDEX["pot"]
        env_layers = [
            pot_layer.astype(jnp.uint8),
            (maze == OBJECT_TO_INDEX["wall"]).astype(jnp.uint8),
            (maze == OBJECT_TO_INDEX["onion_pile"]).astype(jnp.uint8),
            jnp.zeros_like(maze, dtype=jnp.uint8),  # tomato pile (unused)
            (maze == OBJECT_TO_INDEX["plate_pile"]).astype(jnp.uint8),
            (maze == OBJECT_TO_INDEX["goal"]).astype(jnp.uint8),
            jnp.zeros_like(maze, dtype=jnp.uint8),  # 16 onions in pot — skipped for minimal impl.
            jnp.zeros_like(maze, dtype=jnp.uint8),
            jnp.zeros_like(maze, dtype=jnp.uint8),
            jnp.zeros_like(maze, dtype=jnp.uint8),
            jnp.zeros_like(maze, dtype=jnp.uint8),
            jnp.zeros_like(maze, dtype=jnp.uint8),
            (maze == OBJECT_TO_INDEX["plate"]).astype(jnp.uint8),
            (maze == OBJECT_TO_INDEX["onion"]).astype(jnp.uint8),
            jnp.zeros_like(maze, dtype=jnp.uint8),
            jnp.ones_like(maze, dtype=jnp.uint8)
            * ((self.max_steps - state.time) < URGENCY_CUTOFF),
            ]

        obs = jnp.transpose(
            jnp.concatenate([pos_layers, dir_layers, jnp.stack(env_layers)], axis=0), (1, 2, 0)
        )
        return {"agent_0": obs}

    # ------------------------------------------------------------------
    # Boilerplate API helpers
    # ------------------------------------------------------------------

    def _fixed_vec(self, key: str) -> chex.Array:
        idx = jnp.asarray(self.layout[key], dtype=jnp.uint32)
        return jnp.stack([idx % self.width, idx // self.width], axis=-1)

    def is_terminal(self, state: State) -> chex.Array:
        return (state.time >= self.max_steps) | state.terminal

    def action_space(self, agent_id="") -> spaces.Discrete:  # noqa: D401
        return spaces.Discrete(len(self.action_set), dtype=jnp.uint32)

    def observation_space(self):  # noqa: D401
        return spaces.Box(0, 255, self.obs_shape)

    def state_space(self):  # noqa: D401
        h, w = self.height, self.width
        return spaces.Dict(
            {
                "agent_pos": spaces.Box(0, max(h, w), (2,), dtype=jnp.uint32),
                "agent_dir_idx": spaces.Discrete(4),
                "goal_pos": spaces.Box(0, max(h, w), (self.layout["goal_idx"].shape[0], 2), dtype=jnp.uint32),
                "maze_map": spaces.Box(
                    0,
                    255,
                    (w + self.agent_view_size, h + self.agent_view_size, 3),
                    dtype=jnp.uint8,
                ),
                "time": spaces.Discrete(self.max_steps),
                "terminal": spaces.Discrete(2),
            }
        )

    @property
    def name(self):  # noqa: D401
        return f"{self.layout_name}_single"
