# overcooked_single.py
from .overcooked import Overcooked, Actions, DIR_TO_VEC, OBJECT_INDEX_TO_VEC
import jax.numpy as jnp

def _park_dummy_outside(state, env):
    """
    Put agent 1 on the padded-wall corner (0, 0) and
    scrub whatever tile it used to occupy.
    """
    old_x, old_y = state.agent_pos[1]                    # where dummy spawned
    h = env.obs_shape[1]
    pad = (state.maze_map.shape[0] - h) // 2

    # clear old tile
    state = state.replace(
        maze_map = state.maze_map.at[pad + old_y, pad + old_x, :]
        .set(OBJECT_INDEX_TO_VEC[1])          # 'empty' vec
    )

    # move dummy to (0,0) (a wall tile in the padded ring)
    dummy_pos = jnp.array([0, 0], dtype=jnp.uint32)
    state = state.replace(
        agent_pos   = state.agent_pos.at[1].set(dummy_pos),
        agent_dir   = state.agent_dir .at[1].set(DIR_TO_VEC[0]),
        agent_dir_idx = state.agent_dir_idx.at[1].set(0),
    )
    return state


class OvercookedSingle(Overcooked):
    """
    Single-player wrapper: agent_0 plays, agent_1 is hidden off-board.
    """

    # leave num_agents = 2  (don’t break vmaps inside Overcooked!)

    # ---------- public API ----------
    def reset(self, key):
        obs, state = super().reset(key)
        state = _park_dummy_outside(state, self)

        # rebuild obs because we just changed the map
        obs = self.get_obs(state)

        return {"agent_0": obs["agent_0"]}, state

    def step_env(self, key, state, action):
        a0 = action["agent_0"] if isinstance(action, dict) else action
        actions_full = {"agent_0": a0, "agent_1": Actions.stay}

        obs, state, rew, done, info = super().step_env(key, state, actions_full)
        state = _park_dummy_outside(state, self)         # keep it parked

        obs  = {"agent_0": obs["agent_0"]}
        rew  = {"agent_0": rew["agent_0"]}
        done = {"agent_0": done["agent_0"], "__all__": done["__all__"]}

        return obs, state, rew, done, info

    # sugar
    step = step_env
