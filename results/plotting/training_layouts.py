#!/usr/bin/env python3
"""
Plot average *success* or *reward* for the MARL continual-learning benchmark.

Metric semantics
----------------
success : curves are divided by per-environment baseline avg_rewards
          (0 = random agent, 1 = baseline, >1 = out-performing baseline)
reward  : raw reward curves, no normalisation

Usage (examples)
----------------
# success (default)
python plot_avg.py --metric success --data_root results ...

# reward
python plot_avg.py --metric reward --data_root results ...
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import json

# Import utilities from the utils package
try:
    # Try relative import first (when imported as a module)
    from .utils import (
        collect_runs, setup_figure, add_task_boundaries, 
        setup_task_axes, smooth_and_ci, save_plot, finalize_plot,
        CRIT, METHOD_COLORS
    )
except ImportError:
    # Fall back to absolute import (when run as a script)
    from results.plotting.utils import (
        collect_runs, setup_figure, add_task_boundaries, 
        setup_task_axes, smooth_and_ci, save_plot, finalize_plot,
        CRIT, METHOD_COLORS
    )

# Standard colors for different methods
# METHOD_COLORS = {
#     'EWC': '#12939A', 'MAS': '#FF6E54', 'A-GEM': '#FFA600',
#     'L2': '#003F5C', 'PackNet': '#BC5090', 'FT': '#58508D', 
# }

METHOD_COLORS = {
    'EWC':     '#1f77b4',  # blue
    'MAS':     "#FF320E",  # orange
    'A-GEM':   '#2ca02c',  # green
    'L2':      "#7a0000",  # red
    'PackNet': "#F35BB6",  # purple
    'FT':      '#FFA600',  # brown
}

LAYOUT_COLORS = {
    "easy_levels": '#1f77b4',
    "medium_levels": "#FF320E",
    "hard_levels": '#2ca02c',
    'same_size_levels': '#1f77b4', 
    'same_size_padded': "#FF320E",
}

LAYOUT_LABELS = {
    "easy_levels":        "Easy levels",
    "medium_levels":      "Medium levels",
    "hard_levels":        "Hard levels",
    "same_size_levels":   "Same size levels",
    "same_size_padded":   "Same size padded levels",
}




def parse_args():
    """Parse command line arguments for the training plot script."""
    p = argparse.ArgumentParser(description="Plot training metrics for MARL continual-learning benchmark")
    p.add_argument('--data_root', required=True, help="Root directory for data")
    p.add_argument('--algo', required=True, help="Algorithm name")
    p.add_argument('--arch', required=True, help="Architecture name")
    p.add_argument('--methods', required=True, help="Method names to plot")
    p.add_argument('--layouts', nargs='+', required=True, help="Layout categories to plot")
    p.add_argument("--cl_setting", required=True, choices=["DI", "TI"], help="Task-incremental or Domain-incremental")
    p.add_argument('--strategy', required=True, help="Training strategy")
    p.add_argument('--seq_len', type=int, required=True, help="Sequence length")
    p.add_argument('--steps_per_task', type=float, default=1e7, help="Steps per task")
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help="Seeds to include")
    p.add_argument('--sigma', type=float, default=1.5, help="Smoothing parameter")
    p.add_argument('--confidence', type=float, default=0.9, choices=[0.9, 0.95, 0.99], help="Confidence level")
    p.add_argument('--metric', choices=['reward', 'soup'], default='soup', help="Metric to plot")
    p.add_argument('--plot_name', default=None, help="Custom plot name")
    p.add_argument('--legend_anchor', type=float, default=0.87, help="Legend anchor position")
    return p.parse_args()


def load_series(fp: Path) -> np.ndarray:
    """
    Load a time series from a file.
    
    Args:
        fp: Path to the file (.json or .npz)
        
    Returns:
        numpy array containing the time series data
        
    Raises:
        ValueError: If the file has an unsupported extension
    """
    if fp.suffix == '.json':
        return np.array(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == '.npz':
        return np.load(fp)['data'].astype(float)
    raise ValueError(f'Unsupported file suffix: {fp.suffix}')

def collect_runs_on_layout(base, algo, method, strategy, seq_len, seeds, layouts, cl_setting):
    """
    Collect run data for training plots.
    
    Args:
        base: Base directory for data
        algo: Algorithm name
        method: Method name
        strat: Strategy name
        seq_len: Sequence length
        seeds: List of seeds to collect
        layouts: easy, medium, hard, same_size, padded 
        
    Returns:
        Tuple of (data_array, environment_names)
    """
    folder = base / algo / method / f"{strategy}_{seq_len}" / cl_setting / layouts
    print(folder)
    env_names, per_seed = [], []

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        files = sorted(sd.glob(f"training_reward.json"))
        if not files:
            continue

        # first pass → env name order
        if not env_names:
            suffix = f"_reward"
            env_names = [f.name.split('_', 1)[1].rsplit(suffix, 1)[0]
                        for f in files]

        arrs = [load_series(f) for f in files]
        L = max(map(len, arrs))
        padded = [np.pad(a, (0, L - len(a)), constant_values=np.nan)
                 for a in arrs]

        per_seed.append(np.nanmean(padded, axis=0))

    if not per_seed:
        raise RuntimeError(f'No data for method {method}')

    N = max(map(len, per_seed))
    data = np.vstack([np.pad(a, (0, N - len(a)), constant_values=np.nan)
                     for a in per_seed])
    return data, env_names



def plot():
    """
    Main plotting function for training metrics.

    Collects data for each method, creates a plot with method curves,
    adds task boundaries, and saves the plot.
    """
    args = parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    # Calculate total steps and set up figure
    total_steps = args.seq_len * args.steps_per_task
    width = min(max(args.seq_len, 8), 14)
    fig, ax = setup_figure(width=width, height=4)

    # Dictionary to store data for each method
    layouts_data = {}

    # Collect data for each method
    for layout in args.layouts:
        data, env_names = collect_runs_on_layout(
            data_root, args.algo, args.methods,
            args.strategy, args.seq_len, args.seeds, layout, args.cl_setting
        )
        layouts_data[layout] = data

        # Calculate smoothed mean and confidence interval
        mu, ci = smooth_and_ci(data, args.sigma, args.confidence)

        # Plot the method curve
        x = np.linspace(0, total_steps, len(mu))
        color = LAYOUT_COLORS.get(layout)
        label = LAYOUT_LABELS.get(layout, layout)
        ax.plot(x, mu, label=label, color=color)
        ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.1)

    # Add task boundaries
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    add_task_boundaries(ax, boundaries)

    # Set up task axes (primary and secondary x-axes)
    setup_task_axes(ax, boundaries, args.seq_len)

    # Finalize plot with labels, limits, and legend
    finalize_plot(
        ax,
        xlabel='Environment Steps',
        ylabel=f'Average Return',
        xlim=(0, total_steps),
        ylim=(0, None),
        legend_loc='lower center',
        legend_bbox_to_anchor=(0.5, args.legend_anchor),
        legend_ncol=len(args.layouts)
    )

    # Save the plot
    out_dir = Path(__file__).resolve().parent.parent / 'ti_plots'
    stem = args.plot_name or f"avg_norm_{args.metric}"
    save_plot(fig, out_dir, stem)

    # Display the plot
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    plot()