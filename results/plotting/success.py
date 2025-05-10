#!/usr/bin/env python3
"""
Plot average success for the new MARL continual‑learning benchmark.
Assumptions
-----------
* Runs have been downloaded with the provided download script and live in
  data/<algo>/<cl_method>/<strategy>_<seq_len>/seed_<seed>.csv.
* Each CSV has either
    – a single column that measures success (pass it via --metric), or
    – several task‑specific columns that start with "task_" (default).  The
      script averages those per row.
* Every run was logged at equal temporal resolution (one row == one update
  step).  The script pads missing rows with NaN so seeds of unequal length
  still line up.

Usage (example)
---------------
python plot_avg_success_new_benchmark.py \
  --algo marl_algo --methods EWC MAS FT \
  --cl_method EWC MAS FT --strategy ordered --seq_len 6 \
  --seeds 1 2 3 4 5 --plot_name my_plot
"""

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

CRIT = {0.9: 1.833, 0.95: 1.96, 0.99: 2.576}
COLORS = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
    "#777777",
    "#FF8C00",
    "#917113",
]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--algo", required=True)
    p.add_argument("--methods", nargs="+", required=True,
                   help="Folder names of CL methods (e.g. EWC MAS)")
    p.add_argument("--strategy", default="random")
    p.add_argument("--seq_len", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--metric", default=None,
                   help="Column that represents success. If omitted, averages task_* columns.")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="Gaussian smoothing sigma")
    p.add_argument("--confidence", type=float, default=0.95, choices=[0.9, 0.95, 0.99])
    p.add_argument("--plot_name", default="avg_success")
    p.add_argument("--legend_anchor", type=float, default=0.0,
                   help="Y‑offset for legend (0 == tight to plot)")
    return p.parse_args()


def load_csv(path: Path, metric: str | None) -> np.ndarray:
    df = pd.read_csv(path)
    if metric:
        cols = [metric]
    else:
        cols = [c for c in df.columns if c.startswith("task_")]
        if not cols:
            raise ValueError(f"{path} has no task_* columns. Pass --metric instead.")
    return np.nanmean(df[cols].to_numpy(float), axis=1)  # shape (timesteps,)


def collect_runs(root: str, algo: str, method: str, strategy: str, seq_len: int,
                 seeds: List[int], metric: str | None) -> np.ndarray:
    rows = []
    base_dir = Path(__file__).resolve().parent.parent
    for seed in seeds:
        p = base_dir / Path(root) / algo / method / f"{strategy}_{seq_len}" / f"seed_{seed}.csv"
        if not p.exists():
            print(f"missing {p}")
            continue
        rows.append(load_csv(p, metric))
    if not rows:
        raise RuntimeError(f"No runs for {method}")
    # pad to max length with NaN for alignment
    max_len = max(len(r) for r in rows)
    data = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        data[i, : len(r)] = r
    return data  # shape (n_seeds, timesteps)


def plot(args: argparse.Namespace) -> None:
    n_methods = len(args.methods)
    fig, axs = plt.subplots(n_methods, 1, sharex=True,
                            figsize=(12, 4 + 3 * n_methods))
    if n_methods == 1:
        axs = [axs]

    for idx, method in enumerate(args.methods):
        data = collect_runs(args.data_root, args.algo, method, args.strategy,
                            args.seq_len, args.seeds, args.metric)
        mean = gaussian_filter1d(np.nanmean(data, axis=0), sigma=args.sigma)
        std = gaussian_filter1d(np.nanstd(data, axis=0), sigma=args.sigma)
        ci = CRIT[args.confidence] * std / np.sqrt(data.shape[0])
        x = np.arange(len(mean))

        color = COLORS[idx % len(COLORS)]
        ax = axs[idx]
        ax.plot(x, mean, color=color, label=method.upper())
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.25)
        ax.set_ylabel("Average Success")
        ax.set_title(method.upper())
        ax.set_ylim(0, 1)  # Adjust if your metric uses another scale

    axs[-1].set_xlabel("Update Steps")
    axs[-1].legend(loc="lower center", bbox_to_anchor=(0.5, args.legend_anchor),
                   ncol=min(3, n_methods), fancybox=True, shadow=True)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{args.plot_name}.png")
    plt.savefig(f"plots/{args.plot_name}.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot(parse())
