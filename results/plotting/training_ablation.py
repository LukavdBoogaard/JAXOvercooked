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
from typing import List, Tuple

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

METHOD_COLORS = {
    "Baseline": '#2B4C7E',
    "No Task ID": '#F28E2B',
    "No Multi-head": '#E15759',
    "No Layer Norm": '#76B7B2',
    "No Shared Backbone": '#59A14F',
    "CNN": '#EDC948'
}
ablations_to_labels = {
    "no-use_task_id": "No Task ID",
    "no-use_multihead": "No Multi-head",
    "no-use_layer_norm": "No Layer Norm",
    "no-shared_backbone": "No Shared Backbone",
    "use_cnn": "CNN",
    "baseline": "Baseline"
}
def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == ".json":
        return np.asarray(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == ".npz":
        return np.load(fp)["data"].astype(float)
    raise ValueError(fp)


def collect_runs(base: Path, algo: str, method: str, arch: str, strat: str,
                seq_len: int, seeds: List[int], metric: str, cl: str) -> Tuple[np.ndarray, List[str]]:
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
    folder = base / algo / cl / f"{strat}_{seq_len}" / method
    env_names, per_seed = [], []

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        files = sorted(sd.glob(f"*training_{metric}.*"))
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
        raise RuntimeError(f'No data for method {method}')

    N = max(map(len, per_seed))
    data = np.vstack([np.pad(a, (0, N - len(a)), constant_values=np.nan)
                     for a in per_seed])
    return data, env_names


def parse_args():
    """Parse command line arguments for the training plot script."""
    p = argparse.ArgumentParser(description="Plot training metrics for MARL continual-learning benchmark")
    p.add_argument('--data_root', required=True, help="Root directory for data")
    p.add_argument('--algo', required=True, help="Algorithm name")
    p.add_argument('--arch', required=True, help="Architecture name")
    p.add_argument('--cl', required=True, help="cl method name")
    p.add_argument('--methods', nargs='+', required=True, help="Method names to plot")
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
    width = min(12, 14)
    fig, ax = setup_figure(width=10, height=4)

    # Dictionary to store data for each method
    method_data = {}

    # Collect data for each method
    for method in args.methods:
        data, env_names = collect_runs(
            data_root, args.algo, method, args.arch,
            args.strategy, args.seq_len, args.seeds, args.metric, args.cl
        )
        method_data[method] = data

        # Calculate smoothed mean and confidence interval
        mu, ci = smooth_and_ci(data, args.sigma, args.confidence)

        # Plot the method curve
        x = np.linspace(0, total_steps, len(mu))
        label = ablations_to_labels[method]
        color = METHOD_COLORS.get(label)
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
        ylabel=f'IPPO Score Normalized',
        xlim=(0, total_steps),
        ylim=(0, None),
        legend_loc='lower center',
        legend_bbox_to_anchor=(0.5, args.legend_anchor),
        legend_ncol=len(args.methods)
    )

    # Save the plot
    out_dir = Path(__file__).resolve().parent.parent / 'ti_plots'
    stem = args.plot_name or f"avg_norm_{args.metric}"
    save_plot(fig, out_dir, stem)

    # Display the plot
    import matplotlib.pyplot as plt
    plt.show()


def plot_pairwise():
    """
    For each ablation method, create a plot comparing it to the baseline.
    """
    args = parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    total_steps = args.seq_len * args.steps_per_task
    out_dir = Path(__file__).resolve().parent.parent / 'ti_plots'
    out_dir.mkdir(exist_ok=True)

    baseline_key = 'baseline'
    ablation_keys = [m for m in args.methods if m != baseline_key]

    # Load baseline data once
    baseline_data, env_names = collect_runs(
        data_root, args.algo, baseline_key, args.arch,
        args.strategy, args.seq_len, args.seeds, args.metric, args.cl
    )
    baseline_mu, baseline_ci = smooth_and_ci(baseline_data, args.sigma, args.confidence)

    for ablation_key in ablation_keys:
        ablation_data, _ = collect_runs(
            data_root, args.algo, ablation_key, args.arch,
            args.strategy, args.seq_len, args.seeds, args.metric, args.cl
        )
        ablation_mu, ablation_ci = smooth_and_ci(ablation_data, args.sigma, args.confidence)

        # New figure for each comparison
        fig, ax = setup_figure(width=10, height=4)
        x = np.linspace(0, total_steps, len(baseline_mu))

        for key, mu, ci in [
            (baseline_key, baseline_mu, baseline_ci),
            (ablation_key, ablation_mu, ablation_ci)
        ]:
            label = ablations_to_labels[key]
            color = METHOD_COLORS[label]
            ax.plot(x, mu, label=label, color=color)
            ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.1)

        # Add task markers, labels, legend
        boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
        add_task_boundaries(ax, boundaries)
        setup_task_axes(ax, boundaries, args.seq_len)
        finalize_plot(
            ax,
            xlabel='Environment Steps',
            ylabel='IPPO Score Normalized',
            xlim=(0, total_steps),
            ylim=(0, None),
            legend_loc='lower center',
            legend_bbox_to_anchor=(0.5, args.legend_anchor),
            legend_ncol=2
        )

        # Save to file
        stem = args.plot_name or f"{ablations_to_labels[baseline_key]}_vs_{ablations_to_labels[ablation_key]}"
        save_plot(fig, out_dir, stem.lower().replace(" ", "_"))

        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == '__main__':
    plot()
    plot_pairwise()