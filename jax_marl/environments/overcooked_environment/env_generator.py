#!/usr/bin/env python
"""Random Overcooked layout generator + visualisers (refactored).

Key changes compared with the original implementation
-----------------------------------------------------
1. Magic characters have been replaced by descriptive tile constants
   (FLOOR, WALL, GOAL, ONION_PILE, PLATE_PILE, POT, AGENT).
2. Variable and function names have been made more explicit for
   readability.
3. ``wall_density`` now measures the *total* fraction of cells that are
   unpassable to the agent. Interactive tiles (goal, pot, onion‐pile,
   plate‐pile) are placed *before* extra walls are added so that they
   contribute to the density budget.
4. When the generator runs out of empty cells at any placement stage it
   prints a warning and retries with a fresh layout instead of raising
   a cryptic error.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

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
# ─── Symbolic constants ──────────────────────────────────────────────────────
###############################################################################

FLOOR: str = " "
WALL: str = "W"
GOAL: str = "X"
ONION_PILE: str = "O"
PLATE_PILE: str = "B"
POT: str = "P"
AGENT: str = "A"

UNPASSABLE_TILES = {WALL, GOAL, ONION_PILE, PLATE_PILE, POT}
INTERACTIVE_TILES = {GOAL, ONION_PILE, PLATE_PILE, POT}

###############################################################################
# ─── Validation helpers ──────────────────────────────────────────────────────
###############################################################################

def _dfs(i: int, j: int, grid: List[List[str]], visited: List[List[bool]], acc: List[Tuple[int, int, str]]):
    """Depth‑first search used by :func:`evaluate_grid`."""
    if (
            i < 0
            or i >= len(grid)
            or j < 0
            or j >= len(grid[0])
            or visited[i][j]
    ):
        return
    visited[i][j] = True
    acc.append((i, j, grid[i][j]))

    if grid[i][j] not in (FLOOR, AGENT):
        return

    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        _dfs(i + dx, j + dy, grid, visited, acc)


def evaluate_grid(grid_str: str) -> bool:
    """Return ``True`` iff *grid_str* is a valid, solvable Overcooked layout."""
    rows = grid_str.strip().split("\n")
    width = len(rows[0])

    # 1. Rectangularity
    if any(len(r) != width for r in rows):
        return False

    # 2. Required tiles present
    required = [WALL, GOAL, ONION_PILE, PLATE_PILE, POT, AGENT]
    if any(grid_str.count(c) == 0 for c in required) or grid_str.count(AGENT) != 2:
        return False

    # 3. Proper borders (walls or interactives)
    border_ok = {WALL, GOAL, PLATE_PILE, ONION_PILE, POT}
    for y, row in enumerate(rows):
        if y in (0, len(rows) - 1) and any(ch not in border_ok for ch in row):
            return False
        if row[0] not in border_ok or row[-1] not in border_ok:
            return False

    # 4. Interactive tiles must have at least one adjacent free tile (FLOOR or AGENT)
    grid = [list(r) for r in rows]
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch in (AGENT, GOAL, ONION_PILE, PLATE_PILE, POT):
                if all(
                        grid[i + dx][j + dy] not in (FLOOR, AGENT)
                        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
                ):
                    return False

    # 5. Each agent must be able to reach every interactive tile *or* they must be
    #    able to reach a shared wall cell adjacent to both agents.
    agents = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row) if ch == AGENT]

    def reach(start: Tuple[int, int]):
        visited = [[False] * width for _ in rows]
        acc: List[Tuple[int, int, str]] = []
        _dfs(start[0], start[1], grid, visited, acc)
        found = {tile: False for tile in INTERACTIVE_TILES}
        for _, _, c in acc:
            if c in found:
                found[c] = True
        return acc, found

    acc1, found1 = reach(agents[0])
    acc2, found2 = reach(agents[1])

    if all(found1.values()) and all(found2.values()):
        return True  # both can independently reach everything

    # Otherwise at least one agent must be able to reach every interactive tile
    combined_reach = {k: found1[k] or found2[k] for k in found1}
    if not all(combined_reach.values()):
        return False

    pos1 = {(x, y) for x, y, _ in acc1}
    pos2 = {(x, y) for x, y, _ in acc2}

    # Look for a wall adjacent to both reachable regions (a potential hand‑off)
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch == WALL:
                adj_to_1 = any((i + dx, j + dy) in pos1 for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                adj_to_2 = any((i + dx, j + dy) in pos2 for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                if adj_to_1 and adj_to_2:
                    return True
    return False

###############################################################################
# ─── Conversion helper ───────────────────────────────────────────────────────
###############################################################################

def layout_grid_to_dict(grid_str: str) -> FrozenDict:
    """Convert *grid_str* to the JAX‑MARL FrozenDict layout representation."""
    rows = grid_str.strip().split("\n")
    height, width = len(rows), len(rows[0])
    keys = [
        "wall_idx",
        "agent_idx",
        "goal_idx",
        "plate_pile_idx",
        "onion_pile_idx",
        "pot_idx",
    ]
    symbol_to_key = {
        WALL: "wall_idx",
        AGENT: "agent_idx",
        GOAL: "goal_idx",
        PLATE_PILE: "plate_pile_idx",
        ONION_PILE: "onion_pile_idx",
        POT: "pot_idx",
    }

    layout_dict = {k: [] for k in keys}
    layout_dict.update(height=height, width=width)

    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            flat_idx = width * i + j
            if ch in symbol_to_key:
                layout_dict[symbol_to_key[ch]].append(flat_idx)
            # Interactive tiles count as walls for Overcooked's pathing
            if ch in INTERACTIVE_TILES | {WALL}:
                layout_dict["wall_idx"].append(flat_idx)

    # Convert to JAX arrays
    for k in keys:
        layout_dict[k] = jnp.array(layout_dict[k])

    return FrozenDict(layout_dict)

###############################################################################
# ─── Generator ───────────────────────────────────────────────────────────────
###############################################################################

def _random_empty_cell(grid: List[List[str]], rng: random.Random) -> Optional[Tuple[int, int]]:
    """Return a random (row, col) index of a FLOOR cell or ``None`` if none exist."""
    empties = [
        (i, j)
        for i in range(1, len(grid) - 1)
        for j in range(1, len(grid[0]) - 1)
        if grid[i][j] == FLOOR
    ]
    if not empties:
        return None
    return rng.choice(empties)


def _place_tiles(
        grid: List[List[str]],
        tile_symbol: str,
        count: int,
        rng: random.Random,
) -> bool:
    """Place *count* copies of *tile_symbol* on random FLOOR cells.

    Returns ``True`` on success, ``False`` if not enough empty space was available.
    """
    for _ in range(count):
        cell = _random_empty_cell(grid, rng)
        if cell is None:
            return False
        i, j = cell
        grid[i][j] = tile_symbol
    return True


def generate_random_layout(
        *,
        num_agents: int = 2,
        height_rng: Tuple[int, int] = (5, 10),
        width_rng: Tuple[int, int] = (5, 10),
        wall_density: float = 0.15,
        seed: Optional[int] = None,
        max_attempts: int = 2000,
):
    """Generate and return a random solvable Overcooked layout.

    The procedure is:
    1. Draw random width/height.
    2. Add an outer border of walls.
    3. Place interactive tiles (goal, pot, onion pile, plate pile).
    4. Compute how many *additional* walls are needed so that the total
       fraction of unpassable internal cells equals ``wall_density``.
    5. Place those walls.
    6. Finally place the agents.

    The process is retried up to ``max_attempts`` times if any stage runs
    out of empty cells or the resulting grid fails the solvability check.
    """
    rng = random.Random(seed)

    for attempt in range(1, max_attempts + 1):
        height = rng.randint(*height_rng)
        width = rng.randint(*width_rng)

        # Initialise grid with FLOOR
        grid = [[FLOOR for _ in range(width)] for _ in range(height)]

        # Outer walls
        for i in range(height):
            grid[i][0] = grid[i][-1] = WALL
        for j in range(width):
            grid[0][j] = grid[-1][j] = WALL

        # 1. Interactive tiles -------------------------------------------------
        # Up to two of each interactive type
        for symbol in (GOAL, POT, ONION_PILE, PLATE_PILE):
            copies = rng.randint(1, 2)
            if not _place_tiles(grid, symbol, copies, rng):
                print(f"[Attempt {attempt}] Not enough space for {symbol}. Retrying…")
                break  # go to next attempt
        else:  # executed if the loop *didn't* break: all good so far
            # 2. Additional walls so that density matches ----------------------
            internal_cells = (height - 2) * (width - 2)
            current_unpassable = sum(
                1
                for i in range(1, height - 1)
                for j in range(1, width - 1)
                if grid[i][j] in UNPASSABLE_TILES
            )
            target_unpassable = int(round(wall_density * internal_cells))
            additional_walls_needed = max(0, target_unpassable - current_unpassable)

            if not _place_tiles(grid, WALL, additional_walls_needed, rng):
                print(f"[Attempt {attempt}] Could not reach desired wall density. Retrying…")
                continue  # next attempt

            # 3. Agents --------------------------------------------------------
            if not _place_tiles(grid, AGENT, num_agents, rng):
                print(f"[Attempt {attempt}] Not enough space for agents. Retrying…")
                continue

            # Convert to string and validate -----------------------------------
            grid_str = "\n".join("".join(row) for row in grid)
            if evaluate_grid(grid_str):
                return grid_str, layout_grid_to_dict(grid_str)

            print(f"[Attempt {attempt}] Generated layout not solvable. Retrying…")

    raise RuntimeError(
        f"Failed to generate a solvable layout in {max_attempts} attempts."
    )

###############################################################################
# ─── Matplotlib preview ─────────────────────────────────────────────────────
###############################################################################

_TILE_COLOUR = {
    WALL: (0.5, 0.5, 0.5),
    GOAL: (0.1, 0.1, 0.1),
    ONION_PILE: (1, 0.9, 0.2),
    PLATE_PILE: (0.2, 0.2, 0.8),
    POT: (0.9, 0.2, 0.2),
    AGENT: (0.1, 0.7, 0.3),
    FLOOR: (1, 1, 1),
}

def mpl_show(grid_str: str, title: str | None = None):
    rows = grid_str.strip().split("\n")
    height, width = len(rows), len(rows[0])
    img = np.zeros((height, width, 3))
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            img[y, x] = _TILE_COLOUR[ch]

    fig, ax = plt.subplots(figsize=(width / 2, height / 2))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="black", lw=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

###############################################################################
# ─── Overcooked viewer ──────────────────────────────────────────────────────
###############################################################################

def _crop_to_grid(state, view_size: int):
    pad = view_size - 1  # 5→4 because map has +1 outer wall
    return state.maze_map[pad:-pad, pad:-pad, :]

def oc_show(layout: FrozenDict):
    env = Overcooked(layout=layout, layout_name="random_gen", random_reset=False)
    _, state = env.reset(jax.random.PRNGKey(0))
    grid = np.asarray(_crop_to_grid(state, env.agent_view_size))
    vis = OvercookedVisualizer()
    vis.render_grid(grid, tile_size=TILE_PIXELS, agent_dir_idx=state.agent_dir_idx)
    vis.show(block=True)

###############################################################################
# ─── CLI ────────────────────────────────────────────────────────────────────
###############################################################################

def main(argv=None):
    parser = argparse.ArgumentParser("Random Overcooked layout generator")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--num_agents", type=int, default=2, help="number of agents in layout")
    parser.add_argument("--height-min", type=int, default=6, help="minimum layout height")
    parser.add_argument("--height-max", type=int, default=10, help="maximum layout height")
    parser.add_argument("--width-min", type=int, default=6, help="minimum layout width")
    parser.add_argument("--width-max", type=int, default=10, help="maximum layout width")
    parser.add_argument("--wall-density", type=float, default=0.25, help="fraction of unpassable internal cells")
    parser.add_argument("--show", action="store_true", help="preview with matplotlib")
    parser.add_argument("--oc", action="store_true", help="open JAX‑MARL Overcooked viewer")
    parser.add_argument("--save", action="store_true", help="save PNG to assets/screenshots/generated/")
    args = parser.parse_args(argv)

    grid_str, layout = generate_random_layout(
        num_agents=args.num_agents,
        height_rng=(args.height_min, args.height_max),
        width_rng=(args.width_min, args.width_max),
        wall_density=args.wall_density,
        seed=args.seed,
    )
    print(grid_str)

    if args.show:
        mpl_show(grid_str, "Random kitchen")

    if args.oc:
        oc_show(layout)

    if args.save:
        env = Overcooked(layout=layout, layout_name="generated", random_reset=False)
        _, state = env.reset(jax.random.PRNGKey(args.seed or 0))
        grid_arr = np.asarray(_crop_to_grid(state, env.agent_view_size))
        vis = OvercookedVisualizer()
        img = vis._render_grid(grid_arr, tile_size=TILE_PIXELS, agent_dir_idx=state.agent_dir_idx)

        out_dir = Path(__file__).parent.parent.parent.parent / "assets" / "screenshots" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"gen_{args.seed or 'rand'}.png"
        Image.fromarray(img).save(out_dir / file_name)
        print("Saved generated layout to", out_dir / file_name)


if __name__ == "__main__":
    main()
