import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import json
import glob
import numpy as np
from typing import Sequence
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import jax.numpy as jnp
import seaborn as sns

import os, re, json, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from scipy.stats import t


# ---------- helpers ------------------------------------------------
def load_json_array(fname: Path) -> np.ndarray:
    with open(fname, "r") as f:
        return np.asarray(json.load(f), dtype=np.float32)

def natural_sort(files: List[Path]) -> List[Path]:
    """
    Sort files by the leading integer in their filename (0_xxx.json, 1_xxx.json, …)
    """
    def key(p: Path):
        m = re.match(r"(\d+)_", p.name)
        return int(m.group(1)) if m else 1e9
    return sorted(files, key=key)

def split_into_chunks(arr: np.ndarray, n_chunks: int) -> List[np.ndarray]:
    """
    Evenly split *arr* into n_chunks (the last chunk gets the remainder).
    """
    base = len(arr) // n_chunks
    chunks = [arr[i*base:(i+1)*base] for i in range(n_chunks-1)]
    chunks.append(arr[(n_chunks-1)*base:])            # remainder → last chunk
    return chunks

def build_R(task_traces: List[np.ndarray],
            n_points: int = 5) -> np.ndarray:
    """
    task_traces[j] is the full evaluation array for task j.
    Returns R with shape (T+1, T).
    """
    T = len(task_traces)
    R = np.full((T+1, T), np.nan, dtype=np.float32)

    # split every trace into T chunks once, keep for reuse
    chunked = [split_into_chunks(trace, T) for trace in task_traces]

    # baseline (row 0) = mean of first *n_points* of chunk 0
    for j in range(T):
        R[0, j] = np.mean(chunked[j][0][:n_points])

    # after training task i  (row i+1)
    for i in range(T):
        for j in range(T):
            R[i+1, j] = np.mean(chunked[j][i][-n_points:])   # last n_points
    return R


def compute_bwt_matrix(R: np.ndarray) -> np.ndarray:
    assert R.shape[0] == R.shape[1] + 1
    T = R.shape[1]
    bwt = np.full((T, T), np.nan)
    for i in range(T-1):
        for j in range(i+1, T):
            bwt[i, j] = R[j+1, i] - R[i+1, i]
    return bwt

CUT_OFFS = [5, 10, 15, 20, 25]      # evaluate after these many tasks
EPS      = 1e-6   

def compute_nbwt_matrix(R: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    NBWT[i, j] = (R[j+1, i] – R[i+1, i]) / R[i+1, i]
                 for every j > i, provided R[i+1, i] > eps.
    Invalid entries are set to NaN so they drop out of np.nanmean.
    """
    assert R.shape[0] == R.shape[1] + 1
    T = R.shape[1]
    nbwt = np.full((T, T), np.nan, dtype=np.float32)

    for i in range(T - 1):
        denom = R[i + 1, i]         # score immediately after finishing task i
        if denom <= eps:
            continue                # model never really learned task i
        for j in range(i + 1, T):
            nbwt[i, j] = (R[j + 1, i] - denom) / denom
    return nbwt


def compute_single_nbwt(nbwt_matrix: np.ndarray) -> float:
    if np.all(np.isnan(nbwt_matrix)):
        return np.nan
    return float(np.nanmean(nbwt_matrix))



def compute_single_bwt(bwt_matrix: np.ndarray) -> float:
    """Return the average of all valid (non‑NaN) entries of *bwt_matrix*."""
    if np.all(np.isnan(bwt_matrix)):
        return np.nan
    return float(np.nanmean(bwt_matrix))


def mean_and_count(mat: np.ndarray) -> tuple[float, int]:
    """Return (nan-aware mean, number of finite entries)."""
    mask  = ~np.isnan(mat)
    count = int(mask.sum())
    mean  = float(np.nanmean(mat)) if count else np.nan
    return mean, count
def mean_and_count(mat: np.ndarray) -> tuple[float, int]:
    """Return (nan-aware mean, number of finite entries)."""
    mask  = ~np.isnan(mat)
    count = int(mask.sum())
    mean  = float(np.nanmean(mat)) if count else np.nan
    return mean, count
# ───────────────────────────── CONFIG ───────────────────────────── #
ROOT      = Path("/home/luka/repo/JAXOvercooked/results/ablation_data/ippo/Online EWC/random_5")
SEQUENCES = ["random_25"]
LEVELS    = []
SEEDS     = [1, 2, 3, 4]
FACTORS = ["factor 1", "factor 2", "factor 3", "factor 4", "factor 5"]
factors_to_ablations = {
    "factor 1": "use_task_id",
    "factor 2": "use_multihead",
    "factor 3": "use_layer_norm",
    "factor 4": "shared_backbone",
    "factor 5": "no-use_cnn",
}
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
# ─────────────────────────────────────────────────────────────────── #

 

    

# ---------- aggregation & plotting --------------------------------
def aggregate_fwt_bwt(run_root: Path,
                      n_points: int = 5,
                      plot: bool = True):
    """
    Loop over METHODS × SEQUENCES, average FWT and BWT over SEEDS.
    """
    R_list, bwt_list, single_bwt_list, nbwt_list, single_nbwt_list = [], [], [],[], []

    bwt_results = {}
    nbwt_results = {}

    # Baseline
    for factor in FACTORS:
        path = ROOT / factor / factors_to_ablations[factor]
        single_bwt_list = []
        single_nbwt_list = []
        for seed in SEEDS:
            run_dir = path / f"seed_{seed}"
            json_files = natural_sort(run_dir.glob("*_reward.json"))
            json_files = [f for f in json_files if not f.name.startswith("training_reward")]

            if not json_files:
                print(f"No reward files in {run_dir}")
                continue

            task_traces = [load_json_array(f) for f in json_files]
            R = build_R(task_traces, n_points=n_points)
            bwt_mat = compute_bwt_matrix(R)
            nbwt_mat = compute_nbwt_matrix(R)

            nbwt_list.append(nbwt_mat)
            single_bwt_list.append(compute_single_bwt(bwt_mat))
            mean_nbwt, cnt_nbwt = mean_and_count(nbwt_mat)
            single_nbwt_list.append(mean_nbwt)

        if single_bwt_list:
            bwt_results[factors_to_ablations[factor]] = (
                float(np.nanmean(single_bwt_list)),
                float(np.nanstd(single_bwt_list))
            )
            nbwt_results[factors_to_ablations[factor]] =(
                    float(np.nanmean(single_nbwt_list)),
                    float(np.nanstd(single_nbwt_list))
                )   
            

    # Ablations
    for factor in FACTORS:
        path = ROOT / factor / factors_to_ablations_2[factor]
        single_bwt_list = []
        single_nbwt_list = []

        for seed in SEEDS:
            run_dir = path / f"seed_{seed}"
            json_files = natural_sort(run_dir.glob("*_reward.json"))
            json_files = [f for f in json_files if not f.name.startswith("training_reward")]

            if not json_files:
                print(f"No reward files in {run_dir}")
                continue

            task_traces = [load_json_array(f) for f in json_files]
            R = build_R(task_traces, n_points=n_points)
            bwt_mat = compute_bwt_matrix(R)
            nbwt_mat = compute_nbwt_matrix(R)
            single_bwt_list.append(compute_single_bwt(bwt_mat))
            mean_nbwt, cnt_nbwt = mean_and_count(nbwt_mat)
            single_nbwt_list.append(mean_nbwt)

            if single_bwt_list:
                label = ablations_to_labels.get(factors_to_ablations_2[factor], factors_to_ablations_2[factor])
                bwt_results[label] = (
                    float(np.nanmean(single_bwt_list)),
                    float(np.nanstd(single_bwt_list))
            )
                nbwt_results[label] = (
                    float(np.nanmean(single_nbwt_list)),
                    float(np.nanstd(single_nbwt_list))
                )
    
    # Collect BWT results into a DataFrame
    bwt_rows = []
    # Add baseline as "Baseline"
    if "use_task_id" in bwt_results:
        bwt_rows.append({"method": "Baseline", "mean": bwt_results["use_task_id"][0], "std": bwt_results["use_task_id"][1]})

    # Add ablations
    for label in ["No Task ID", "No Multi-head", "No Layer Norm", "No Shared Backbone", "CNN"]:
        if label in bwt_results:
            bwt_rows.append({"method": label, "mean": bwt_results[label][0], "std": bwt_results[label][1]})

    bwt_df = pd.DataFrame(bwt_rows)

    # create a dataframe for nbwt results
    nbwt_rows = []
    # Add baseline as "Baseline"
    if "use_task_id" in nbwt_results:
        nbwt_rows.append({"method": "Baseline", "mean": nbwt_results["use_task_id"][0], "std": nbwt_results["use_task_id"][1]})
    # Add ablations
    for label in ["No Task ID", "No Multi-head", "No Layer Norm", "No Shared Backbone", "CNN"]:
        if label in nbwt_results:
            nbwt_rows.append({"method": label, "mean": nbwt_results[label][0], "std": nbwt_results[label][1]})
    nbwt_df = pd.DataFrame(nbwt_rows)

    # Now you can use bwt_df for your bar chart code
    print(nbwt_df)
    # print(bwt_df)

    # df = pd.DataFrame(rows)
    if bwt_df.empty:
        raise RuntimeError("No matching data found; check paths/arguments.")

    # # aggregate: mean + 95 % CI
    # agg = (bwt_df.groupby(["method"])["score"]
    #        .agg(["mean", "count", "std"])
    #        .reset_index())
    # agg["ci95"] = agg.apply(
    #     lambda r: r["std"] / np.sqrt(r["count"]) * t.ppf(0.975, r["count"] - 1)
    #     if r["count"] > 1 else np.nan, axis=1)


    # Compute 95% CI for each row (assuming n=number of seeds)
    n_seeds = len(SEEDS)
    bwt_df["ci95"] = bwt_df["std"] / np.sqrt(n_seeds) * t.ppf(0.975, n_seeds - 1) if n_seeds > 1 else np.nan
    nbwt_df["ci95"] = nbwt_df["std"] / np.sqrt(n_seeds) * t.ppf(0.975, n_seeds - 1) if n_seeds > 1 else np.nan



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

    methods = list(palette.keys())
    # Ensure the order matches the palette
    # plot_df = bwt_df.set_index("method").reindex(methods).reset_index()
    plot_df = nbwt_df.set_index("method").reindex(methods).reset_index()
    means = plot_df["mean"].values
    ci95s = plot_df["ci95"].values

    bar_w = 0.5
    x = np.arange(len(methods))

    bars = ax.bar(x, means, bar_w, yerr=ci95s, capsize=5,
                  color=[palette[m] for m in methods], alpha=0.95, zorder=3)

    # Remove x-tick labels (no names under bars)
    ax.yaxis.grid(True, linestyle='-', alpha=0.5, zorder=0)
    ax.xaxis.grid(True, linestyle='-', alpha=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(methods))
    ax.set_ylabel(f"Backward Transfer")
    ax.set_xlabel("Ablation")

    # Add legend with method names and corresponding colors, outside the plot
    legend_handles = [plt.matplotlib.patches.Patch(color=palette[m], label=m) for m in methods]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=11)

    plt.tight_layout()
    out = ROOT / 'plots'
    out.mkdir(exist_ok=True)
    stem = "ablation_bwt_barchart"
    plt.savefig(out / f"{stem}.png", bbox_inches='tight')
    plt.savefig(out / f"{stem}.pdf", bbox_inches='tight')
    plt.show()


    # # Print results
    # for label, (mean, std) in bwt_results.items():
    #     print(f"{label}: {mean:.4f} ± {std:.4f}")

    # for label, (mean, std) in nbwt_results.items():
    #     print(f"{label}: {mean:.4f} ± {std:.4f}")

            
# ─────────────────────────── entry point ───────────────────────────
if __name__ == "__main__":
    aggregate_fwt_bwt(ROOT, n_points=5, plot=True)


