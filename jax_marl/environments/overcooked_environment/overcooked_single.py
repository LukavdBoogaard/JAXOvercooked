# overcooked_single.py  (replace the previous wrapper)

import jax.numpy as jnp

from .overcooked import Overcooked, Actions, DIR_TO_VEC, OBJECT_INDEX_TO_VEC


def _park_dummy(state, env):
    """
    Move dummy agent 1 to the outer padded wall (0,0) and
    clear its previous tile so it never blocks gameplay.
    """
    dummy_old_x, dummy_old_y = state.agent_pos[1]
    h = env.obs_shape[1]
    pad = (state.maze_map.shape[0] - h) // 2
    empty_vec = OBJECT_INDEX_TO_VEC[1]  # 'empty'

    # scrub where the dummy used to be
    state = state.replace(
        maze_map=state.maze_map.at[pad + dummy_old_y,
                 pad + dummy_old_x, :].set(empty_vec)
    )
    # park it at the corner wall
    state = state.replace(
        agent_pos=state.agent_pos.at[1].set(jnp.array([0, 0], jnp.uint32)),
        agent_dir=state.agent_dir.at[1].set(DIR_TO_VEC[0]),
        agent_dir_idx=state.agent_dir_idx.at[1].set(0),
    )
    return state


class OvercookedSingle(Overcooked):
    """
    One-chef façade around the stock 2-chef Overcooked.
    • agent_0 plays,
    • agent_1 is parked off-board and gets zero-obs/zero-reward.
    """

    def reset(self, key):
        obs, state = super().reset(key)
        state = _park_dummy(state, self)

        obs = self.get_obs(state)
        obs["agent_1"] = jnp.zeros_like(obs["agent_0"])  # dummy view
        return obs, state

    def step_env(self, key, state, actions):
        # ignore whatever the trainer sends for agent_1
        a0 = actions["agent_0"] if isinstance(actions, dict) else actions
        full_act = {"agent_0": a0, "agent_1": Actions.stay}

        obs, state, rew, done, info = super().step_env(key, state, full_act)
        state = _park_dummy(state, self)

        obs = self.get_obs(state)
        obs["agent_1"] = jnp.zeros_like(obs["agent_0"])

        # zero-out dummy reward so it never pollutes logs
        rew["agent_1"] = 0.0
        return obs, state, rew, done, info

    # optional sugar
    step = step_env
