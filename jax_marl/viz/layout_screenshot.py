#!/usr/bin/env python3
import os
import jax
import numpy as np
from PIL import Image

from jax_marl.environments import Overcooked
from jax_marl.environments.overcooked_environment.layouts import (
    hard_layouts,
    medium_layouts,
    easy_layouts,
)
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer, TILE_PIXELS


def save_start_states(grouped_layouts: dict, base_dir: str = "../../assets/screenshots"):
    key = jax.random.PRNGKey(0)
    vis = OvercookedVisualizer()

    for difficulty, layouts in grouped_layouts.items():
        out_dir = os.path.join(base_dir, difficulty)
        os.makedirs(out_dir, exist_ok=True)

        for name, layout in layouts.items():
            key, subkey = jax.random.split(key)
            env = Overcooked(layout=layout)
            _, state = env.reset(subkey)

            grid = np.asarray(state.maze_map)
            img = vis._render_grid(
                grid,
                tile_size=TILE_PIXELS,
                highlight_mask=None,
                agent_dir_idx=state.agent_dir_idx,
                agent_inv=state.agent_inv
            )

            path = os.path.join(out_dir, f"{name}.png")
            Image.fromarray(img).save(path)
            print(f"Saved {path}")


if __name__ == "__main__":
    grouped = {
        "easy": easy_layouts,
        "medium": medium_layouts,
        "hard": hard_layouts
    }
    save_start_states(grouped)
