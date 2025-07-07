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
from scipy import stats

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

# ---------- transfer matrices -------------------------------------
def compute_fwt_matrix(R: np.ndarray) -> np.ndarray:
    assert R.shape[0] == R.shape[1] + 1
    T = R.shape[1]
    fwt = np.full((T, T), np.nan)
    for i in range(T-1):
        for j in range(i+1, T):
            fwt[i, j] = R[i+1, j] - R[0, j]
    return fwt

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


def mean_ci95(values: list[float]) -> tuple[float, float]:
    arr  = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(arr)
    # ignore NaNs when counting df
    n    = np.sum(~np.isnan(arr))
    if n < 2:
        return float(mean), np.nan
    sem  = stats.sem(arr, nan_policy="omit")
    ci95 = 1.96 * sem            # normal approx.
    return float(mean), float(ci95)

# ───────────────────────────── CONFIG ───────────────────────────── #
ROOT      = Path("/home/luka/repo/JAXOvercooked/results/layout_data_raw/ippo/Online EWC/random_5/TI")
METHODS   = ["easy_levels", "hard_levels", "medium_levels"]
SEEDS     = [1, 2, 3, 4]
# ─────────────────────────────────────────────────────────────────── #
LABELS = {"easy_levels":   "Easy",
          "medium_levels": "Medium",
          "hard_levels":   "Hard"}


# ---------- aggregation & plotting --------------------------------
def aggregate_fwt_bwt(run_root: Path,
                      n_points: int = 5,
                      plot: bool = True):
    """
    
    """
    bwt_rows = []
    nbwt_rows = []
    # -------------- inside aggregate_fwt_bwt --------------------------
    for method in METHODS:                          # "easy_levels" … "hard_levels"
        single_bwt_per_seed  = []
        single_nbwt_per_seed = []

        for seed in SEEDS:
            run_dir = run_root / method / f"seed_{seed}"
            json_files = natural_sort(run_dir.glob("*_reward.json"))
            json_files = [f for f in json_files
                        if not f.name.startswith("training_reward")]
            if not json_files:
                print(f"No reward files in {run_dir}")
                continue

            task_traces = [load_json_array(f) for f in json_files]
            R        = build_R(task_traces, n_points=n_points)
            bwt_mat  = compute_bwt_matrix(R)
            nbwt_mat = compute_nbwt_matrix(R)

            single_bwt_per_seed.append(compute_single_bwt(bwt_mat))
            single_nbwt_per_seed.append(compute_single_nbwt(nbwt_mat))

        # ---------- store per-difficulty aggregate ------------------- #
        mean_bwt,  ci95_bwt  = mean_ci95(single_bwt_per_seed)
        mean_nbwt, ci95_nbwt = mean_ci95(single_nbwt_per_seed)

        bwt_rows.append(
            {"difficulty": method,          # <- keep raw key
            "mean":       mean_bwt,
            "ci95":       ci95_bwt,
            "n":          len(single_bwt_per_seed)}
        )
        nbwt_rows.append(
            {"difficulty": method,
            "mean":       mean_nbwt,       # <- the NBWT mean
            "ci95":       ci95_nbwt,
            "n":          len(single_nbwt_per_seed)}
        )

    # ------------------------------------------------------------------
    # DataFrames
    bwt_df  = pd.DataFrame(bwt_rows)
    nbwt_df = pd.DataFrame(nbwt_rows)

    # plotting (example for BWT)
    palette   = {"easy_levels": '#4E79A7',
                "medium_levels": '#F28E2B',
                "hard_levels": '#E15759'}
    seq_order = ["easy_levels", "medium_levels", "hard_levels"]

    # plot_df = bwt_df.set_index("difficulty").reindex(seq_order).reset_index()
    plot_df = nbwt_df.set_index("difficulty").reindex(seq_order).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(seq_order))
    bar_w = 0.55

    ax.bar(
        x,
        plot_df["mean"].values,
        bar_w,
        yerr=plot_df["ci95"].values,
        capsize=5,
        color=[palette[s] for s in seq_order],
        alpha=0.95,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in seq_order])   # "Easy Medium Hard"
    ax.set_ylabel("Backward Transfer (BWT)")
    ax.yaxis.grid(True, linestyle='-', alpha=0.5, zorder=0)

    plt.tight_layout()
    out = run_root / "plots";  out.mkdir(exist_ok=True)
    stem = "ewc_nbwt_difficulty_barchart"
    plt.savefig(out / f"{stem}.png",  bbox_inches="tight")
    plt.savefig(out / f"{stem}.pdf",  bbox_inches="tight")
    plt.show()

        # ── optional plots ────────────────────────────────────────────
        # if plot:
        #     save_dir = run_root / method / "heatmaps"
        #     save_dir.mkdir(exist_ok=True, parents=True)

        #     # for name, mat in [("bwt", bwt_mean), ("nbwt", nbwt_mean)]:
        #     #     plt.figure(figsize=(16,10))
        #     #     sns.heatmap(mat, annot=False, fmt=".2f",
        #     #                 cmap="coolwarm", center=0,
        #     #                 vmin=-1, vmax=0.1,
        #     #                 xticklabels=[f"Task {j}" for j in range(mat.shape[1])],
        #     #                 yticklabels=[f"Task {i}" for i in range(mat.shape[0])])
        #     #     plt.title(f"{method} – {seq} – mean {name.upper()} over {len(R_list)} seeds")
        #     #     plt.xlabel("Task B"), plt.ylabel("Task A")
        #     #     plt.tight_layout()
        #     #     plt.savefig(save_dir / f"{name}_mean_{method}.png")
        #     #     plt.close()

        #     for name, mat in [("bwt", bwt_mean), ("nbwt", nbwt_mean)]:
        #         fig, ax = plt.subplots(figsize=(14, 10))          # a bit narrower than 16 × 10

        #         hm = sns.heatmap(
        #             mat,
        #             ax=ax,
        #             annot=True,                   # show the numbers
        #             fmt=".2f",
        #             cmap="coolwarm",
        #             center=0,
        #             vmin=-1,
        #             vmax=0.1,
        #             xticklabels=list(range(mat.shape[1])),   # 0, 1, 2 …
        #             yticklabels=list(range(mat.shape[0])),
        #             cbar_kws={
        #                 "fraction": 0.035,        # width of the colour-bar (smaller = thinner)
        #                 "pad": 0.02,              # space between heat-map and colour-bar
        #             },
        #         )

        #         # Bigger tick labels
        #         ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
        #         ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

        #         cbar = hm.collections[0].colorbar
        #         cbar.ax.tick_params(labelsize=14)

        #         # Bigger annotation numbers
        #         for text in ax.texts:
        #             text.set_fontsize(12)

        #         # ax.set_title(f"{method} – {seq} – mean {name.upper()} over {len(R_list)} seeds",
        #         #             fontsize=16, pad=12)
        #         ax.set_xlabel("Task B", fontsize=14)
        #         ax.set_ylabel("Task A", fontsize=14)

        #         plt.tight_layout()
        #         fig.savefig(save_dir / f"{name}_mean_{method}_annotated.png", dpi=300)
        #         plt.close(fig)

#────────────────────────────────────────────────────────────── #
    # report numbers
    print()
    


# ─────────────────────────── entry point ───────────────────────────
if __name__ == "__main__":
    aggregate_fwt_bwt(ROOT, n_points=2, plot=True)
