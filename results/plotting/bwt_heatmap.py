# bwt_heatmap.py

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from results.download.common import cli, want
from plasticity import compute_bwt

def load_curve(path):
    """Load a JSON reward curve or .npz array."""
    if path.endswith(".json"):
        return np.loadtxt(path, delimiter=",")
    else:
        return np.load(path)["arr_0"]

def build_perf_matrix(curves, seq_len, steps_per_task):
    """
    curves: list of 1D arrays, one per task, of length L
    returns (T+1)xT matrix: row 0 is before any training,
            row j+1 is after training on task j.
    """
    T = len(curves)
    L = curves[0].shape[0]
    total = seq_len * steps_per_task
    t = np.linspace(0, total, L, endpoint=False)
    M = np.full((T+1, T), np.nan)
    for i, arr in enumerate(curves):
        # before any training on task i
        M[0, i] = np.nanmean(arr[t < i*steps_per_task])
        # after training on task j, for each j
        for j in range(T):
            mask = (t >= j*steps_per_task) & (t < (j+1)*steps_per_task)
            M[j+1, i] = np.nanmean(arr[mask])
    return M

def main():
    args = cli()
    out = args.output       # e.g. "data/"
    seq_len = args.seq_length
    spt = args.steps_per_task  # ensure this flag exists in common.cli()

    # For each run (algo/method/arch/strat/seed) that you downloaded:
    for algo in os.listdir(out):
      for method in os.listdir(f"{out}/{algo}"):
        for arch in os.listdir(f"{out}/{algo}/{method}"):
          for strat in os.listdir(f"{out}/{algo}/{method}/{arch}"):
            for seed in os.listdir(f"{out}/{algo}/{method}/{arch}/{strat}"):
              run_id = f"{algo}/{method}/{arch}/{strat}/{seed}"
              folder = os.path.join(out, run_id)
              if not want(run_id, args): 
                  continue

              # find all per-task eval files
              paths = sorted(glob.glob(f"{folder}/*_reward.json"))
              if not paths: 
                  continue

              # load seed's curves and build its perf matrix
              curves = [load_curve(p) for p in paths]
              perf_mat = build_perf_matrix(curves, seq_len, spt)

              # compute seed-specific BWT matrix
              seed_bwt = compute_bwt(perf_mat)

              # accumulate into a list
              try:
                  all_bwt = np.dstack((all_bwt, seed_bwt))
              except NameError:
                  all_bwt = seed_bwt[..., np.newaxis]

    # average across seeds (axis=2)
    avg_bwt = np.nanmean(all_bwt, axis=2)

    # now plot the averaged BWT heatmap
    T = avg_bwt.shape[0]
    vmax = np.nanmax(np.abs(avg_bwt))
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(avg_bwt, vmin=-vmax, vmax=+vmax, cmap="coolwarm", aspect="auto")
    # annotate cells
    for (i,j), val in np.ndenumerate(avg_bwt):
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(T))
    ax.set_yticks(np.arange(T))
    ax.set_xticklabels([f"Task {j}" for j in range(T)], rotation=45, ha="right")
    ax.set_yticklabels([f"Step {i+1}" for i in range(T)])  # steps 1…T
    ax.set_xlabel("Evaluated Task")
    ax.set_ylabel("After Training Step")
    plt.colorbar(im, ax=ax, label="Backward Transfer")
    plt.title("Average BWT Across Seeds")
    plt.tight_layout()
    os.makedirs("heatmap_images", exist_ok=True)
    plt.savefig("heatmap_images/average_bwt_heatmap.png")
    plt.close()

if __name__ == "__main__":
    main()
