#!/usr/bin/env python3
"""
Plot average success for the new MARL continual-learning benchmark with refined styling.

Updates in this version (May 2025):
1. Top axis now shows *environment indexes* centered via secondary_xaxis for exact alignment.
2. Vertical task-boundary lines remain between tasks.
3. Bottom x-axis ticks aligned with boundaries and formatted in scientific notation.
4. Figure width reduced; tick label fonts enlarged for readability.

Usage: unchanged CLI interface from previous script.
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# seaborn-darkgrid style without default grid overlay
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['axes.grid'] = False

# critical z-values for confidence intervals
CRIT = {0.9: 1.833, 0.95: 1.96, 0.99: 2.576}

METHOD_COLORS = {
    'EWC': '#12939A', 'MAS': '#FF6E54', 'AGEM': '#FFA600',
    'L2': '#58508D', 'PackNet': '#BC5090', 'ReDo': '#003F5C', 'CBP': '#2F4B7C'
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--algo', required=True)
    p.add_argument('--methods', nargs='+', required=True)
    p.add_argument('--strategy', required=True)
    p.add_argument('--seq_len', type=int, required=True)
    p.add_argument('--steps_per_task', type=float, default=8e6)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--confidence', type=float, default=0.95, choices=[0.9, 0.95, 0.99])
    p.add_argument('--plot_name', default='avg_success')
    p.add_argument('--legend_anchor', type=float, default=0.0)
    return p.parse_args()


def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == '.json':
        return np.array(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == '.npz':
        return np.load(fp)['data'].astype(float)
    raise ValueError(f'Unsupported file suffix: {fp.suffix}')


def collect_runs(base: Path, algo: str, method: str, strat: str, seq_len: int, seeds: List[int]):
    folder = base / algo / method / f"{strat}_{seq_len}"
    env_names, per_seed = [], []
    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists(): continue
        files = sorted(sd.glob("*_success.*"))
        if not files: continue
        if not env_names:
            env_names = [f.name.split('_',1)[1].rsplit('_success',1)[0] for f in files]
        arrs = [load_series(f) for f in files]
        max_len = max(map(len, arrs))
        padded = [np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in arrs]
        per_seed.append(np.nanmean(padded, axis=0))
    if not per_seed:
        raise RuntimeError(f'No data for method: {method}')
    N = max(map(len, per_seed))
    data = np.vstack([np.pad(a, (0, N - len(a)), constant_values=np.nan) for a in per_seed])
    return data, env_names


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent.parent / args.data_root

    width = max(10, args.seq_len * 1.2)
    fig, ax = plt.subplots(figsize=(width, 4))

    total_steps = args.seq_len * args.steps_per_task

    # plot each method's average success with CI shading
    for method in args.methods:
        data, _ = collect_runs(base, args.algo, method, args.strategy, args.seq_len, args.seeds)
        mu = gaussian_filter1d(np.nanmean(data, axis=0), sigma=args.sigma)
        sd = gaussian_filter1d(np.nanstd(data, axis=0), sigma=args.sigma)
        ci = CRIT[args.confidence] * sd / np.sqrt(data.shape[0])
        x = np.linspace(0, total_steps, len(mu))
        color = METHOD_COLORS.get(method.upper())
        ax.plot(x, mu, label=method, color=color)
        ax.fill_between(x, mu - ci, mu + ci, color=color, alpha=0.15)

    # draw vertical task boundaries
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    for b in boundaries[1:-1]:
        ax.axvline(b, color='gray', ls='--', lw=0.5)

    # bottom x-axis: boundaries in scientific notation
    ax.set_xticks(boundaries)
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='both', labelsize=10)

    # top axis: exact centering with secondary_xaxis
    secax = ax.secondary_xaxis('top')
    mids = [(boundaries[i] + boundaries[i+1]) / 2.0 for i in range(args.seq_len)]
    secax.set_xticks(mids)
    secax.set_xticklabels([str(i+1) for i in range(args.seq_len)], fontsize=10)
    secax.tick_params(axis='x', length=0)

    # labels and legend
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Average Reward')
    ax.set_xlim(0, total_steps)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, args.legend_anchor), ncol=3)

    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / Path('plots')
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / f"{args.plot_name}.png")
    plt.savefig(out_dir / f"{args.plot_name}.pdf")
    plt.show()


if __name__ == '__main__':
    main()