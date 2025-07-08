#!/usr/bin/env python3
"""
Plot *per‑seed* training curves for the MARL continual‑learning benchmark.

This version is a simplification of the original ``plot_avg.py``: instead of
averaging across multiple methods, it accepts **one** method (``--method``) and
loops over the requested seeds, drawing one curve per seed so you can inspect
variance directly.

Metric semantics
----------------
success : curves are divided by per‑environment baseline avg_rewards
          (0 = random agent, 1 = baseline, >1 = out‑performing baseline)
reward  : raw reward curves, no normalisation

Usage (examples)
----------------
# success (default)
python plot_seed_curves.py --metric success \
       --data_root results --method EWC --algo ippo --arch cnn \
       --strategy seq --seq_len 10 --seeds 1 2 3 4 5

# reward with custom colours
python plot_seed_curves.py --metric reward --method A-GEM \
       --colormap tab20
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from results.plotting.utils.common import load_series

# Import utilities from the utils package
try:
    # Try relative import first (when imported as a module)
    from .utils import (
        collect_runs, setup_figure, add_task_boundaries,
        setup_task_axes, smooth_and_ci, save_plot, finalize_plot,
    )
except ImportError:
    # Fall back to absolute import (when run as a script)
    from results.plotting.utils import (
        collect_runs, setup_figure, add_task_boundaries,
        setup_task_axes, smooth_and_ci, save_plot, finalize_plot,
    )


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command‑line arguments for per‑seed training plot."""
    p = argparse.ArgumentParser(
        description="Plot per‑seed training curves for MARL continual‑learning benchmark"
    )

    # Data location & experiment description
    p.add_argument('--data_root', required=True, help="Root directory for data")
    p.add_argument('--algo',      required=True, help="Algorithm name (e.g. ippo)")
    # p.add_argument('--arch',      required=True, help="Architecture name")
    p.add_argument('--method',    required=True, help="Single method to plot")
    p.add_argument('--strategy',  required=True, help="Training strategy")

    # Sequence / task setup
    p.add_argument('--seq_len', type=int, required=True, help="Number of tasks in sequence")
    p.add_argument('--steps_per_task', type=float, default=1e7, help="Environment steps per task")

    # Seeds
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                   help="Seeds to plot individually")

    # Plotting hyper‑parameters
    p.add_argument('--sigma', type=float, default=1.5, help="Gaussian smoothing σ")
    p.add_argument('--confidence', type=float, default=0.9, choices=[0.9, 0.95, 0.99],
                   help="Confidence level for shaded region")
    p.add_argument('--metric', choices=['reward', 'success'], default='reward',
                   help="Metric to plot")
    p.add_argument('--plot_name', default=None, help="Custom file name stem for the saved plot")
    p.add_argument('--legend_anchor', type=float, default=0.87, help="Legend anchor y‑pos")
    p.add_argument('--colormap', default='tab10', help="Matplotlib colormap for seeds")

    return p.parse_args()






def collect_runs(base: Path, algo: str, method: str, strat: str,
                seq_len: int, seeds: List[int], metric: str) -> Tuple[np.ndarray, List[str]]:
    """
    Collect run data for training plots.
    
    Args:
        base: Base directory for data
        algo: Algorithm name
        method: Method name
        arch: Architecture name
        strat: Strategy name
        seq_len: Sequence length
        seeds: List of seeds to collect
        metric: Metric to collect ('reward', 'soup', etc.)
        
    Returns:
        Tuple of (data_array, environment_names)
    """
    folder = base / algo / method / f"{strat}_{seq_len}"
    env_names, per_seed = [], []

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        print(sd)
        if not sd.exists():
            continue
        files = sorted(sd.glob(f"*training_reward.*"))
        if not files:
            continue

        # first pass → env name order
        if not env_names:
            suffix = f"_{metric}"
            env_names = [f.name.split('_', 1)[1].rsplit(suffix, 1)[0]
                        for f in files]

        arrs = [load_series(f) for f in files]
        L = max(map(len, arrs))
        padded = [np.pad(a, (0, L - len(a)), constant_values=np.nan)
                 for a in arrs]

        per_seed.append(np.nanmean(padded, axis=0))

    if not per_seed:
        raise RuntimeError(f'No data for method')

    N = max(map(len, per_seed))
    data = np.vstack([np.pad(a, (0, N - len(a)), constant_values=np.nan)
                     for a in per_seed])
    return data, env_names


# -----------------------------------------------------------------------------
# Main plotting routine
# -----------------------------------------------------------------------------

def plot():
    """Create and save the per‑seed training plot."""
    args = parse_args()

    # Resolve paths
    data_root = Path(__file__).resolve().parent.parent / args.data_root
    out_dir   = Path(__file__).resolve().parent.parent / 'ti_plots'

    # Derived quantities
    total_steps = args.seq_len * args.steps_per_task
    width = min(max(args.seq_len, 8), 14)  # heuristic width in inches

    # Set‑up figure / axis
    fig, ax = setup_figure(width=width, height=4)

    # Choose a colour for every seed from the requested colormap
    # cmap = plt.get_cmap(args.colormap, len(args.seeds))
    # palette = {
    #     0: '#4E79A7',
    #     2: '#F28E2B',
    #     3: '#E15759',
    #     4: '#76B7B2',
    #     1: '#59A14F',
    #     5: '#EDC948'
    # }

    palette = {
     0: "#4E79A7",  # blue
     1: "#59A14F",  # green
     2: "#F28E2B",  # orange
     3: "#E15759",  # red
     4: "#76B7B2",  # teal
     5: "#EDC948",  # yellow
}

    # Loop over seeds ---------------------------------------------------------
    for i, seed in enumerate(args.seeds):
        # ``collect_runs`` expects a list of seeds → pass singleton [seed]
        data, _env_names = collect_runs(
            data_root, args.algo, args.method,
            args.strategy, args.seq_len, [seed], args.metric
        )

        # Smooth + confidence interval (CI will be 0 with one run, but keep API)
        mu, ci = smooth_and_ci(data, args.sigma, args.confidence)

        # X‑axis: environment steps
        x = np.linspace(0, total_steps, len(mu))

        # Plot curve & CI
        color = palette.get(i)
        label = f"seed {seed}"
        ax.plot(x, mu, label=label, color=color)
        ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.1)

    # Task boundaries & secondary x‑axis -------------------------------------
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    add_task_boundaries(ax, boundaries)
    setup_task_axes(ax, boundaries, args.seq_len)

    # Final touches -----------------------------------------------------------
    ylabel = 'Average Return'
    finalize_plot(
        ax,
        xlabel='Environment Steps',
        ylabel=ylabel,
        xlim=(0, total_steps),
        ylim=(0, None),
        legend_loc='lower center',
        legend_bbox_to_anchor=(0.5, args.legend_anchor),
        legend_ncol=min(len(args.seeds), 6),
    )

    # Save to disk ------------------------------------------------------------
    stem = args.plot_name or f"{args.algo}_{args.method}_per_seed_{args.metric}_{args.seq_len}"
    save_plot(fig, out_dir, stem)

    plt.show()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    plot()
