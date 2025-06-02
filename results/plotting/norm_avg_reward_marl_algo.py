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
ALGO_COLORS = {
    'CNN': '#12939A', 'FT': '#FF6E54'
}



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--algo', required=True)
    p.add_argument('--archs', nargs='+', required=True)
    # p.add_argument('--methods', nargs='+', required=True)
    p.add_argument('--strategy', required=True)
    p.add_argument('--seq_len', type=int, required=True)
    p.add_argument('--steps_per_task', type=float, default=1e7)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--sigma', type=float, default=1.5)
    p.add_argument('--confidence', type=float, default=0.9,
                   choices=[0.9, 0.95, 0.99])
    p.add_argument('--metric', choices=['reward'], default='reward')
    p.add_argument('--plot_name', default=None)
    p.add_argument('--legend_anchor', type=float, default=0.87)
    p.add_argument('--baseline_file',
                   default='practical_reward_baseline_results.yaml',
                   help="Normalize reward curves by the baseline results from this file")
    return p.parse_args()


def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == '.json':
        return np.array(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == '.npz':
        return np.load(fp)['data'].astype(float)
    raise ValueError(f'Unsupported file suffix: {fp.suffix}')


def collect_runs(base: Path, algo: str, arch: str, strat: str,
                 seq_len: int, seeds: List[int], metric: str,
                 baselines: dict | None):
    folder = base / algo / arch / f"{strat}_{seq_len}"
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

        arrs = [load_series(f) for f in files]
        L = max(map(len, arrs))
        padded = [np.pad(a, (0, L - len(a)), constant_values=np.nan)
                  for a in arrs]

        # success → divide by baseline reward per env
        if metric == 'success':
            if baselines is None:
                raise ValueError("No baseline YAML supplied")
            denom = []
            for nm in env_names:
                b = baselines.get(nm, {}).get('avg_rewards')
                denom.append(np.nan if (b is None or b == 0) else b)
            padded = [padded[i] / denom[i] for i in range(len(padded))]

        per_seed.append(np.nanmean(padded, axis=0))

    if not per_seed:
        raise RuntimeError(f'No data for algo {algo}')

    N = max(map(len, per_seed))
    data = np.vstack([np.pad(a, (0, N - len(a)), constant_values=np.nan)
                      for a in per_seed])
    return data, env_names


def plot():
    args = parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root
    baselines = {}
    if args.metric == 'success':
        with open(Path(__file__).resolve().parent.parent.parent / args.baseline_file) as f:
            baselines = yaml.safe_load(f)

    total_steps = args.seq_len * args.steps_per_task
    width = min(max(args.seq_len, 8), 14)
    fig, ax = plt.subplots(figsize=(width, 4))

    for arch in args.archs:
        data, env_names = collect_runs(data_root, args.algo, arch,
                                       args.strategy, args.seq_len,
                                       args.seeds, args.metric, baselines)
        print(data.shape, data)
        mu = gaussian_filter1d(np.nanmean(data, axis=0), sigma=args.sigma)
        sd = gaussian_filter1d(np.nanstd(data, axis=0), sigma=args.sigma)
        ci = CRIT[args.confidence] * sd / np.sqrt(data.shape[0])
        x = np.linspace(0, total_steps, len(mu))
        color = ALGO_COLORS.get(arch)
        ax.plot(x, mu, label=arch, color=color)
        ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.2)

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
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, args.legend_anchor), ncol=len(args.archs))

    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / 'plots'
    out.mkdir(exist_ok=True)
    stem = args.plot_name or f"avg_norm_{args.metric}"
    plt.savefig(out / f"{stem}.png")
    plt.savefig(out / f"{stem}.pdf")
    plt.show()


if __name__ == '__main__':
    plot()
