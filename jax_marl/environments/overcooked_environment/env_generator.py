#!/usr/bin/env python
"""Random Overcooked layout generator + visualisers.

Usage examples
--------------
```bash
# Print ASCII and preview in matplotlib
python env_generator.py --seed 123 --show

# Same kitchen but display through the official Overcooked viewer (JAX-MARL)
python env_generator.py --seed 123 --oc
```

* `--show`   → matplotlib quick view (no deps beyond Matplotlib)
* `--oc`     → OvercookedViewer view (needs `jax_marl` installed)

Every emitted level passes the original `evaluate_grid` check, so it is
solvable by construction.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from flax.core.frozen_dict import FrozenDict

from jax_marl.environments import Overcooked
from jax_marl.gridworld.grid_viz import TILE_PIXELS
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer


###############################################################################
# ---  validation ----------------------------------------------------------- #
###############################################################################

def _dfs(i, j, G, vis, acc):
    if i < 0 or i >= len(G) or j < 0 or j >= len(G[0]) or vis[i][j]:
        return
    vis[i][j] = True
    acc.append((i, j, G[i][j]))
    if G[i][j] not in (" ", "A"):
        return
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        _dfs(i + dx, j + dy, G, vis, acc)


def evaluate_grid(grid: str) -> bool:
    rows = grid.strip().split("\n")
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        return False
    req = ["W", "X", "O", "B", "P", "A"]
    if any(grid.count(c) == 0 for c in req) or grid.count("A") != 2:
        return False
    border_ok = {"W", "X", "B", "O", "P"}
    for y, r in enumerate(rows):
        if y in (0, len(rows) - 1) and any(ch not in border_ok for ch in r):
            return False
        if r[0] not in border_ok or r[-1] not in border_ok:
            return False
    G = [list(r) for r in rows]
    for i, r in enumerate(G):
        for j, ch in enumerate(r):
            if ch in ("A", "X", "O", "B", "P") and all(
                    G[i + dx][j + dy] not in (" ", "A") for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))):
                return False
    agents = [(i, j) for i, r in enumerate(G) for j, ch in enumerate(r) if ch == "A"]

    def _reach(start):
        vis = [[False] * w for _ in rows]
        acc = []
        _dfs(start[0], start[1], G, vis, acc)
        flags = {e: False for e in ("X", "O", "B", "P")}
        for _, _, c in acc:
            if c in flags:
                flags[c] = True
        return acc, flags

    acc1, f1 = _reach(agents[0])
    acc2, f2 = _reach(agents[1])
    if all(f1.values()) and all(f2.values()):
        return True
    coll = {k: f1[k] or f2[k] for k in f1}
    if not all(coll.values()):
        return False
    pos1, pos2 = {(x, y) for x, y, _ in acc1}, {(x, y) for x, y, _ in acc2}
    for i, r in enumerate(G):
        for j, ch in enumerate(r):
            if ch == "W":
                adj1 = any((i + dx, j + dy) in pos1 for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                adj2 = any((i + dx, j + dy) in pos2 for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                if adj1 and adj2:
                    return True
    return False


###############################################################################
# ---  conversion helper ---------------------------------------------------- #
###############################################################################

def layout_grid_to_dict(grid: str) -> FrozenDict:
    rows = grid.strip().split("\n")
    h, w = len(rows), len(rows[0])
    keys = ["wall_idx", "agent_idx", "goal_idx", "plate_pile_idx", "onion_pile_idx", "pot_idx"]
    sym = {"W": "wall_idx", "A": "agent_idx", "X": "goal_idx",
           "B": "plate_pile_idx", "O": "onion_pile_idx", "P": "pot_idx"}
    lay = {k: [] for k in keys}
    lay.update(height=h, width=w)
    for i, r in enumerate(rows):
        for j, ch in enumerate(r):
            idx = w * i + j
            if ch in sym:
                lay[sym[ch]].append(idx)
            if ch in ("X", "B", "O", "P"):
                lay["wall_idx"].append(idx)
    for k in keys:
        lay[k] = jnp.array(lay[k])
    return FrozenDict(lay)


###############################################################################
# ---  random generator ----------------------------------------------------- #
###############################################################################

def _empty(G, rng):
    empties = [(i, j) for i in range(1, len(G) - 1) for j in range(1, len(G[0]) - 1) if G[i][j] == " "]
    return rng.choice(empties)


def generate_random_layout(height_rng=(5, 10), width_rng=(5, 10), wall_density=0.15,
                           seed: Optional[int] = None, max_attempts=2000):
    rng = random.Random(seed)
    for _ in range(max_attempts):
        h, w = rng.randint(*height_rng), rng.randint(*width_rng)
        G = [[" " for _ in range(w)] for _ in range(h)]
        for i in range(h):
            G[i][0] = G[i][-1] = "W"
        for j in range(w):
            G[0][j] = G[-1][j] = "W"
        internal = [(i, j) for i in range(1, h - 1) for j in range(1, w - 1)]
        for i, j in rng.sample(internal, int(len(internal) * wall_density)):
            G[i][j] = "W"
        # 2 agents are mandatory
        for _ in range(2):
            i, j = _empty(G, rng)
            G[i][j] = "A"
        # up to two of every other interactive tile
        for ch in ("X", "P", "O", "B"):          # delivery, pot, onion-pile, plate-pile
            n = rng.randint(1, 2)                # 1 or 2 of each
            for _ in range(n):
                i, j = _empty(G, rng)
                G[i][j] = ch
        grid = "\n".join("".join(r) for r in G)
        if evaluate_grid(grid):
            return grid, layout_grid_to_dict(grid)
    raise RuntimeError("no solvable layout found in allotted attempts")


###############################################################################
# ---  matplotlib quick preview -------------------------------------------- #
###############################################################################

_COL = {"W": (0.5, 0.5, 0.5), "X": (0.1, 0.1, 0.1), "O": (1, 0.9, 0.2),
        "B": (0.2, 0.2, 0.8), "P": (0.9, 0.2, 0.2), "A": (0.1, 0.7, 0.3), " ": (1, 1, 1)}


def mpl_show(grid: str, title: str | None = None):
    rows = grid.strip().split("\n")
    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 3))
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            img[y, x] = _COL[ch]
    fig, ax = plt.subplots(figsize=(w / 2, h / 2))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="black", lw=0.5)
    ax.set_xticks([]);
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


###############################################################################
# ---  Overcooked viewer ---------------------------------------------------- #
###############################################################################

def _crop_to_grid(state, view_size: int):
    pad = view_size - 1  # 5→4 because map has +1 outer wall
    return state.maze_map[pad:-pad, pad:-pad, :]


def oc_show(layout: FrozenDict):
    env = Overcooked(layout=layout, layout_name="random_gen", random_reset=False)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    grid = np.asarray(_crop_to_grid(state, env.agent_view_size))
    vis = OvercookedVisualizer()
    vis.render_grid(grid, tile_size=TILE_PIXELS, agent_dir_idx=state.agent_dir_idx)
    vis.show(block=True)


###############################################################################
# ---  CLI ------------------------------------------------------------------ #
###############################################################################

def main(argv=None):
    p = argparse.ArgumentParser("Random Overcooked layout generator")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--show", action="store_true", help="preview with matplotlib")
    p.add_argument("--oc", action="store_true", help="open JAX-MARL Overcooked viewer")
    p.add_argument("--save", action="store_true", help="save PNG to assets/screenshots/generated/")
    args = p.parse_args(argv)

    grid, layout = generate_random_layout(seed=args.seed)
    print(grid)

    if args.show:
        mpl_show(grid, "Random kitchen")

    if args.oc:
        oc_show(layout)

    if args.save:
        # use the same OvercookedVisualizer you have in oc_show
        env = Overcooked(layout=layout, layout_name="generated", random_reset=False)
        _, state = env.reset(jax.random.PRNGKey(args.seed or 0))
        grid_arr = np.asarray(_crop_to_grid(state, env.agent_view_size))
        vis = OvercookedVisualizer()
        img = vis._render_grid(grid_arr, tile_size=TILE_PIXELS, agent_dir_idx = state.agent_dir_idx)

        out_dir = Path(__file__).parent.parent.parent.parent / "assets" / "screenshots" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"gen_{args.seed or 'rand'}.png"
        Image.fromarray(img).save(out_dir / file_name)
        print("Saved generated layout to", out_dir / file_name)


if __name__ == "__main__":
    main()
