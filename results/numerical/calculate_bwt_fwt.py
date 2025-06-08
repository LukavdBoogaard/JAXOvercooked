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

# def chunk_list_by(array: list[float], seq_length) -> list[list[float]]:
#     n = len(array) // seq_length
#     return [array[i * n:(i + 1) * n] for i in range(seq_length)]


# def create_matrix(seq_len, chunks):
#     pass



# def compute_fwt(matrix):
#     """
#     Computes the forward transfer for all tasks in a sequence
#     param matrix: a 2D array of shape (num_tasks + 1, num_tasks) where each entry is the performance of the model on the task
#     """
#     # Assert that the matrix has the correct shape
#     assert matrix.shape[0] == matrix.shape[1] + 1, "Matrix must have shape (num_tasks + 1, num_tasks)"

#     num_tasks = matrix.shape[1]

#     fwt_matrix = np.full((num_tasks, num_tasks), np.nan)

#     for i in range(1, num_tasks):
#         for j in range(i):  # j < i
#             before_learning = matrix[0, i]
#             after_task_j = matrix[j + 1, i]
#             fwt_matrix[i, j] = after_task_j - before_learning

#     return fwt_matrix


# def compute_bwt(matrix):
#     """
#     Computes the backward transfer for all tasks in a sequence
#     param matrix: a 2D array of shape (num_tasks + 1, num_tasks) where each entry is the performance of the model on the task
#     """
#     assert matrix.shape[0] == matrix.shape[1] + 1, "Matrix must have shape (num_tasks + 1, num_tasks)"
#     num_tasks = matrix.shape[1]

#     bwt_matrix = jnp.full((num_tasks, num_tasks), jnp.nan)

#     for i in range(num_tasks - 1):
#         for j in range(i + 1, num_tasks):
#             after_j = matrix[j + 1, i]  # performance on task i after learning task j
#             after_i = matrix[i + 1, i]  # performance on task i after learning task i
#             bwt_matrix = bwt_matrix.at[i, j].set(after_j - after_i)

#     return bwt_matrix


# def show_heatmap_bwt(matrix, run_name, save_folder="heatmap_images"):
#     # Ensure the save folder exists
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)

#     bwt_matrix = compute_bwt(matrix)
#     avg_bwt_per_step = np.nanmean(bwt_matrix, axis=0)

#     fig, ax = plt.subplots(figsize=(10, 7))
#     sns.heatmap(bwt_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f",
#                 xticklabels=[f"Task {j}" for j in range(bwt_matrix.shape[1])],
#                 yticklabels=[f"Task {i}" for i in range(bwt_matrix.shape[0])],
#                 cbar_kws={"label": "BWT"})
#     ax.set_title("Progressive Backward Transfer Matrix")
#     ax.set_xlabel("Task B")
#     ax.set_ylabel("Task A")
#     plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

#     # Add average BWT per step below the heatmap
#     for j, val in enumerate(avg_bwt_per_step):
#         if not np.isnan(val):
#             ax.text(j + 0.5, len(avg_bwt_per_step) + 0.2, f"{val:.2f}",
#                     ha='center', va='bottom', fontsize=9, color='black')
#     plt.text(-0.7, len(avg_bwt_per_step) + 0.2, "Avg", fontsize=10, va='bottom', weight='bold')

#     plt.tight_layout()

#     # Save the figure
#     file_path = os.path.join(save_folder, f"{run_name}_bwt_heatmap.png")
#     plt.savefig(file_path)
#     plt.close()


# def show_heatmap_fwt(matrix, run_name, save_folder="heatmap_images"):
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)

#     fwt_matrix = compute_fwt(matrix)
#     avg_fwt_per_step = np.nanmean(fwt_matrix, axis=0)

#     fig, ax = plt.subplots(figsize=(10, 7))
#     sns.heatmap(fwt_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f",
#                 xticklabels=[f"Task {j}" for j in range(fwt_matrix.shape[1])],
#                 yticklabels=[f"Task {i}" for i in range(fwt_matrix.shape[0])],
#                 cbar_kws={"label": "FWT"})
#     ax.set_title("Progressive Forward Transfer Matrix")
#     ax.set_xlabel("Task B")
#     ax.set_ylabel("Task A")

#     plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

#     for j, val in enumerate(avg_fwt_per_step):
#         if not np.isnan(val):
#             ax.text(j + 0.5, len(avg_fwt_per_step) + 0.2, f"{val:.2f}",
#                     ha='center', va='bottom', fontsize=9, color='black')

#     plt.text(-0.7, len(avg_fwt_per_step) + 0.2, "Avg", fontsize=10, va='bottom', weight='bold')

#     plt.tight_layout()

#     file_path = os.path.join(save_folder, f"{run_name}_fwt_heatmap.png")
#     plt.savefig(file_path)
#     plt.close()


import os, re, json, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict

# ───────────────────────────── CONFIG ───────────────────────────── #
ROOT      = Path("/home/luka/repo/JAXOvercooked/results/data/experiment_2/ippo")
METHODS   = ["EWC"]
SEQUENCES = ["random_5"]
SEEDS     = [0, 1, 2, 3, 4]
# ─────────────────────────────────────────────────────────────────── #

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
    print(R)
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
            bwt[i, j] = R[i+1, i] - R[j+1, i]
    return bwt

# ---------- aggregation & plotting --------------------------------
def aggregate_fwt_bwt(run_root: Path,
                      n_points: int = 5,
                      plot: bool = True):
    """
    Loop over METHODS × SEQUENCES, average FWT and BWT over SEEDS.
    """
    for method in METHODS:
        for seq in SEQUENCES:
            R_list, fwt_list, bwt_list = [], [], []

            for seed in SEEDS:
                run_dir = run_root / method / seq / f"seed_{seed}"
                json_files = natural_sort(run_dir.glob("*_reward.json"))
                # ignore training_reward.json
                json_files = [f for f in json_files if not f.name.startswith("training_reward")]

                if not json_files:
                    print(f"No reward files in {run_dir}")
                    continue

                task_traces = [load_json_array(f) for f in json_files]
                R = build_R(task_traces, n_points=n_points)
                R_list.append(R)
                fwt_list.append(compute_fwt_matrix(R))
                bwt_list.append(compute_bwt_matrix(R))

            if not R_list:
                continue

            R_mean = np.nanmean(R_list, axis=0)
            fwt_mean = np.nanmean(fwt_list, axis=0)
            fwt_std  = np.nanstd (fwt_list, axis=0)
            bwt_mean = np.nanmean(bwt_list, axis=0)
            bwt_std  = np.nanstd (bwt_list, axis=0)

            # ── optional plots ────────────────────────────────────────────
            if plot:
                save_dir = run_root / method / seq / "heatmaps"
                save_dir.mkdir(exist_ok=True, parents=True)

                for name, mat in [("fwt", fwt_mean), ("bwt", bwt_mean)]:
                    plt.figure(figsize=(8,6))
                    sns.heatmap(mat, annot=True, fmt=".2f",
                                cmap="coolwarm", center=0,
                                xticklabels=[f"Task {j}" for j in range(mat.shape[1])],
                                yticklabels=[f"Task {i}" for i in range(mat.shape[0])])
                    plt.title(f"{method} – {seq} – mean {name.upper()} over {len(R_list)} seeds")
                    plt.xlabel("Task B"), plt.ylabel("Task A")
                    plt.tight_layout()
                    plt.savefig(save_dir / f"{name}_mean.png")
                    plt.close()

            # ── report numbers ───────────────────────────────────────────
            print(f"\n{method} / {seq}  – averaged over {len(R_list)} seeds")
            print("Mean FWT:\n", fwt_mean)
            print("Std  FWT:\n", fwt_std)
            print("Mean BWT:\n", bwt_mean)
            print("Std  BWT:\n", bwt_std)
            print("Average performance: \n", R_mean)

# ─────────────────────────── entry point ───────────────────────────
if __name__ == "__main__":
    aggregate_fwt_bwt(ROOT, n_points=2, plot=True)
