#!/usr/bin/env python3
"""
Compare MLP vs. CNN for several CL methods with a bar-chart (mean + 95 % CI).

Example
-------
python plot_bar.py --data_root results --algo ippo \
                   --methods EWC MAS L2 --strategy ordered \
                   --seq_len 10 --metric reward
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t

sns.set_theme(style="whitegrid", context="notebook")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="root folder: results/<algo>/<method>/<arch>/strategy_len/seed_*")
    p.add_argument("--algo", required=True)
    p.add_argument("--methods", nargs="+")
    p.add_argument("--strategy", required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--metric", choices=["reward", "success"], default="reward")
    p.add_argument("--seeds", nargs="+", type=int, default=[3])
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == ".json":
        return np.asarray(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == ".npz":
        return np.load(fp)["data"].astype(float)
    raise ValueError(fp)

def split_into_chunks(arr: np.ndarray, n_chunks: int) -> List[np.ndarray]:
    """
    Evenly split *arr* into n_chunks (the last chunk gets the remainder).
    """
    base = len(arr) // n_chunks
    chunks = [arr[i*base:(i+1)*base] for i in range(n_chunks-1)]
    chunks.append(arr[(n_chunks-1)*base:])            # remainder → last chunk
    return chunks


def final_scores(folder: Path, metric: str, seeds: list[int]) -> list[float]:
    scores = []
    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists(): 
            continue
        files = sorted(sd.glob(f"training_reward.*"))
        if not files: 
            continue

        per_task_means = []

        for f in files:
            array = load_series(f)
            chunks = split_into_chunks(array, 5)
            task_means = [np.mean(chunk[-10:]) for chunk in chunks]
            if task_means:
                per_task_means.append(np.mean(task_means))

        if per_task_means:
            scores.append(np.mean(per_task_means)) 
 
    return scores


def ci95(vals: np.ndarray) -> float:
    if len(vals) < 2: return np.nan
    return vals.std(ddof=1) / np.sqrt(len(vals)) * t.ppf(0.975, len(vals) - 1)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    base = root / args.data_root / args.algo / 'Online EWC' / 'random_5'
    rows = []

    factors = ["factor 1", "factor 2", "factor 3", "factor 4", "factor 5"]
    factors_to_ablations = {
            "factor 1": "use_task_id",
            "factor 2": "use_multihead",
            "factor 3": "use_layer_norm",
            "factor 4": "shared_backbone",
            "factor 5": "no-use_cnn",
        }

    baseline_paths = []
    for factor in factors:
        path = base / factor / factors_to_ablations[factor]
        baseline_paths.append(path)
    
    for baseline in baseline_paths:
        vals = final_scores(baseline, args.metric, args.seeds)
        for v in vals: 
            rows.append(dict(method="Baseline", score=v))
    
    factors_to_ablations_2 = {
            "factor 1": "no-use_task_id",
            "factor 2": "no-use_multihead",
            "factor 3": "no-use_layer_norm",
            "factor 4": "no-shared_backbone",
            "factor 5": "use_cnn",
        }
    
    ablations_to_labels = {
            "no-use_task_id": "No Task ID",
            "no-use_multihead": "No Multi-head",
            "no-use_layer_norm": "No Layer Norm",
            "no-shared_backbone": "No Shared Backbone",
            "use_cnn": "CNN"
        }
    
    ablation_paths = []
    for factor in factors:
        path = path = base / factor / factors_to_ablations_2[factor]
        ablation_paths.append(path)
    
    for ablation in ablation_paths:
        vals = final_scores(ablation, args.metric, args.seeds)
        ablation_name = ablations_to_labels[ablation.name]  # gets the last part of the path
        print(ablation_name)
        for v in vals: 
            rows.append(dict(method=ablation_name, score=v))
    

    # for method in args.methods:
    #     for arch in ("MLP", "CNN"):
    #         run_dir = base / method / arch / f"{args.strategy}_{args.seq_len}"
    #         vals = final_scores(run_dir, args.metric, args.seeds)
    #         for v in vals:
    #             rows.append(dict(method=method, arch=arch, score=v))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matching data found; check paths/arguments.")

    # aggregate: mean + 95 % CI
    agg = (df.groupby(["method"])["score"]
           .agg(["mean", "count", "std"])
           .reset_index())
    agg["ci95"] = agg.apply(
        lambda r: r["std"] / np.sqrt(r["count"]) * t.ppf(0.975, r["count"] - 1)
        if r["count"] > 1 else np.nan, axis=1)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    width = 8
    fig, ax = plt.subplots(figsize=(width, 4))

    # Define a bright, popping color palette for each ablation and the baseline
    palette = {
        "Baseline": '#4E79A7',
        "No Task ID": '#F28E2B',
        "No Multi-head": '#E15759',
        "No Layer Norm": '#76B7B2',
        "No Shared Backbone": '#59A14F',
        "CNN": '#EDC948'
    }

    bar_w = 0.5
    x = np.arange(len(palette))

    # Plot a single bar for baseline and each ablation
    methods = list(palette.keys())
    means = [agg.loc[agg.method == m, "mean"].values[0] if m in agg.method.values else np.nan for m in methods]
    ci95s = [agg.loc[agg.method == m, "ci95"].values[0] if m in agg.method.values else np.nan for m in methods]

    bars = ax.bar(x, means, bar_w, yerr=ci95s, capsize=5,
                  color=[palette[m] for m in methods], alpha=0.95)

    # Remove x-tick labels (no names under bars)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(methods))
    ax.set_ylabel(f"Average ISN Score")
    ax.set_xlabel("Ablation")

    # Add legend with method names and corresponding colors, outside the plot
    legend_handles = [plt.matplotlib.patches.Patch(color=palette[m], label=m) for m in methods]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=11)

    plt.tight_layout()
    out = root / 'plots'
    out.mkdir(exist_ok=True)
    stem = "ablation_barchart"
    plt.savefig(out / f"{stem}.png", bbox_inches='tight')
    plt.savefig(out / f"{stem}.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
