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
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['axes.grid'] = False

# CRIT = {0.9: 1.833, 0.95: 1.96, 0.99: 2.576}
CRIT = {0.9: 1, 0.95: 1.96, 0.99: 2.576}


ARCH_COLORS = {
    "positive": '#12939A', 
    "negative": '#FF6E54'
}  


EXPERIMENTS = {
    "arch_multihead": ["use_multihead", "no-use_multihead"],
    "arch_task_id": ["use_task_id", "no-use_task_id"],
    "arch_backbone": ["shared_backbone", "no-shared_backbone"],
    "arch_cnn_mlp": ["CNN", "MLP"],
}


def collect_all_paths(root: Path, separator):
    # Collect all paths to files in the directory tree
    paths = []
    for path in root.rglob('*'):
        if path.is_file():
            relative_path = path.relative_to(root)
            paths.append(str(relative_path).replace('\\', separator))
    return paths
            
def binary_comparison_architectures(tag_1: str, tag_2: str, root: Path):
    # collect the full path to al files 
    all_paths = collect_all_paths(root, separator='/')

    # create the correct strings based on the architecture parameter
    positive_string = f"/{tag_1}" # e.g. "/use_multihead"
    negative_string = f"/{tag_2}" # e.g. "/no-use_multihead"

    # filter the paths for the positive and negative strings
    positive_paths = [p for p in all_paths if positive_string in p]
    negative_paths = [p for p in all_paths if negative_string in p]

    # if no paths are found, raise an error
    if not positive_paths or not negative_paths:
        raise ValueError(f"Could not find paths for {tag_1} or {tag_2}")
    
    # make sure both lists have the same length
    if len(positive_paths) != len(negative_paths):
        max_len = max(len(positive_paths), len(negative_paths))
        positive_paths = positive_paths + [''] * (max_len - len(positive_paths))
        negative_paths = negative_paths + [''] * (max_len - len(negative_paths))
    
    # collect the files from all positive paths
    pos_files = []
    for path in positive_paths:
        p = root / path
        if not p.exists():
            continue
        if 'training_reward' not in p.name:
            continue
        pos_files.append(load_series(p))

    # collect the files from all negative paths
    neg_files = []
    for path in negative_paths:
        p = root / path
        if not p.exists():
            continue
        if 'training_reward' not in p.name:
            continue
        neg_files.append(load_series(p))
    
    print(len(pos_files), "positive files found")
    print(len(neg_files), "negative files found")


    N_pos = max(map(len, pos_files))
    N_neg = max(map(len, neg_files))
    pos_data = np.vstack([np.pad(a, (0, N_pos - len(a)), constant_values=np.nan)
                          for a in pos_files])
    neg_data = np.vstack([np.pad(a, (0, N_neg - len(a)), constant_values=np.nan)
                          for a in neg_files])
    
    return pos_data, neg_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--algo', required=True)
    # p.add_argument('--archs', nargs="+", required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--strategy', required=True)
    p.add_argument('--seq_len', type=int, required=True)
    p.add_argument('--steps_per_task', type=float, default=1e7)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 3])
    p.add_argument('--sigma', type=float, default=1.5)
    p.add_argument('--confidence', type=float, default=0.9,
                   choices=[0.9, 0.95, 0.99])
    p.add_argument('--metric', choices=['reward'], default='reward')
    p.add_argument('--plot_name', default=None)
    p.add_argument('--legend_anchor', type=float, default=0.87)
    # p.add_argument('--baseline_file',
    #                default='practical_reward_baseline_results.yaml',
    #                help="Normalize reward curves by the baseline results from this file")
    return p.parse_args()


def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == '.json':
        series = np.array(json.loads(fp.read_text()), dtype=float)

        # For every value, if it's higher than 2.0, divide by 340.0
        mask = series > 2.0
        series[mask] = series[mask] / 340.0

        return series
    if fp.suffix == '.npz':
        return np.load(fp)['data'].astype(float)
    raise ValueError(f'Unsupported file suffix: {fp.suffix}')


def collect_runs(base: Path, algo: str, method: str, arch_path: Path, strat: str,
                 seq_len: int, seeds: List[int], metric: str,
                 baselines: dict | None):
    folder = base / algo / method / f"{strat}_{seq_len}" / arch_path
    env_names, per_seed = [], []

    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        files = sorted(sd.glob("*_reward.*"))
        if not files:
            continue

        # first pass → env name order
        if not env_names:
            suffix = "_reward"
            env_names = [f.name.split('_', 1)[1].rsplit(suffix, 1)[0]
                         for f in files]

        # Load each file into a numpy array
        arrs = [load_series(f) for f in files]
        # Make sure all arrays are the same length
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

def plot_comparison_tags(args, plot_name: str, tag1: str, tag2: str):
    data_root = Path(__file__).resolve().parent.parent / args.data_root
    total_steps = args.seq_len * args.steps_per_task
    width = min(max(args.seq_len, 12), 14) 
    fig, ax = plt.subplots(figsize=(width, 3.4))

    pos_data, neg_data = binary_comparison_architectures(tag1, tag2, data_root)
    pos_mu = gaussian_filter1d(np.nanmean(pos_data, axis=0), sigma=args.sigma)
    neg_mu = gaussian_filter1d(np.nanmean(neg_data, axis=0), sigma=args.sigma)
    pos_sd = gaussian_filter1d(np.nanstd(pos_data, axis=0), sigma=args.sigma)
    neg_sd = gaussian_filter1d(np.nanstd(neg_data, axis=0), sigma=args.sigma)
    pos_ci = CRIT[args.confidence] * pos_sd / np.sqrt(pos_data.shape[0])
    neg_ci = CRIT[args.confidence] * neg_sd / np.sqrt(neg_data.shape[0])
    x = np.linspace(0, total_steps, len(pos_mu))

    ax.plot(x, pos_mu, label=tag1, color=ARCH_COLORS['positive'])
    ax.plot(x, neg_mu, label=tag2, color=ARCH_COLORS['negative'])
    ax.fill_between(x, pos_mu - pos_ci, pos_mu + pos_ci, color=ARCH_COLORS['positive'], alpha=0.2)
    ax.fill_between(x, neg_mu - neg_ci, neg_mu + neg_ci, color=ARCH_COLORS['negative'], alpha=0.2)

    # vertical lines at task boundaries
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    for b in boundaries[1:-1]:
        ax.axvline(b, color='gray', ls='--', lw=0.5)

    # x-axes
    ax.set_xticks(boundaries)
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    secax = ax.secondary_xaxis('top')
    mids = [(boundaries[i] + boundaries[i + 1]) / 2.0
            for i in range(args.seq_len)]
    secax.set_xticks(mids)
    secax.set_xticklabels(["Task " + str(i + 1) for i in range(args.seq_len)],
                          fontsize=12)
    secax.tick_params(axis='x', length=0)

    # labels & legend
    y_label = 'Average Success' if args.metric == 'success' else 'Average Performance'
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel(y_label)
    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, None)
    ax.legend(loc='lower left')#, bbox_to_anchor=(0.5, args.legend_anchor)) #, ncol=len(args.method))
    # ax.set_title(f"Comparison of {tag1} and {tag2} Architectures", )
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / 'plots'
    out.mkdir(exist_ok=True)
    plt.savefig(out / f"{plot_name}.png")
    plt.savefig(out / f"{plot_name}.pdf")
    plt.show()


def plot():
    args = parse_args()
    # baselines = {}
    # if args.metric == 'success':
    #     with open(Path(__file__).resolve().parent.parent.parent / args.baseline_file) as f:
    #         baselines = yaml.safe_load(f)

    # arch_path = "CNN/shared_backbone/no-use_multihead/use_task_id"
    # data, env_names = collect_runs(
    #     data_root, args.algo, args.method, arch_path, args.strategy, args.seq_len, args.seeds, args.metric, None)
    
    # print(data.shape)

    for arch in EXPERIMENTS.items():
        print(f"Plotting {arch[0]}...")
        plot_comparison_tags(args, arch[0], arch[1][0], arch[1][1])
    


if __name__ == '__main__':
    plot()
