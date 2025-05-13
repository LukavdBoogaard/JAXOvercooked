#!/usr/bin/env python3
"""Download evaluation curves *per environment* for the MARL continual-learning
benchmark and store them one metric per file.

Optimized logic:
1. Discover available evaluation keys per run via `run.history(samples=1)`.
2. Fetch each key's full time series separately, only once.
3. Skip keys whose output files already exist (unless `--overwrite`).
4. Write files in `data/<algo>/<cl_method>/<strategy>_<seq_len>/seed_<seed>/`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import wandb
from wandb.apis.public import Run

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
FORBIDDEN_TAGS = {"TEST", "LOCAL"}
EVAL_PREFIX = "Scaled_returns/evaluation_"
KEY_PATTERN = re.compile(rf"^{re.escape(EVAL_PREFIX)}(\d+)__(.+)_scaled$")
TRAINING_KEY = "Scaled_returns/returned_episode_returns_scaled"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--output", default="data", help="Base folder for output")
    p.add_argument("--format", choices=["json", "npz"], default="json", help="Output file format")
    p.add_argument("--seq_length", type=int, nargs="+", default=[6])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--strategy", choices=["ordered", "random"], default=None)
    p.add_argument("--algos", nargs="+", default=[], help="Filter by alg_name")
    p.add_argument("--cl_methods", nargs="+", default=[], help="Filter by cl_method")
    p.add_argument("--wandb_tags", nargs="+", default=[], help="Require at least one tag")
    p.add_argument("--include_runs", nargs="+", default=[], help="Include runs by substring")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# FILTER
# ---------------------------------------------------------------------------
def want(run: Run, args: argparse.Namespace) -> bool:
    cfg = run.config
    if any(tok in run.name for tok in args.include_runs): return True
    if run.state != "finished": return False
    if args.seeds and cfg.get("seed") not in args.seeds: return False
    if args.algos and cfg.get("alg_name") not in args.algos: return False
    if args.cl_methods and cfg.get("cl_method") not in args.cl_methods: return False
    if args.seq_length and cfg.get("seq_length") not in args.seq_length: return False
    if args.strategy and cfg.get("strategy") != args.strategy: return False
    if 'wandb_tags' in cfg:
        tags = cfg['wandb_tags']['value']
        if args.wandb_tags and not tags.intersection(args.wandb_tags): return False
        if tags.intersection(FORBIDDEN_TAGS) and not tags.intersection(args.wandb_tags): return False
    return True


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def discover_eval_keys(run: Run) -> List[str]:
    """Retrieve & sort eval keys, plus the one training key if present."""
    df = run.history(samples=1)
    # only exact eval keys
    keys = [k for k in df.columns if KEY_PATTERN.match(k)]
    # include training series, if logged
    if TRAINING_KEY in df.columns:
        keys.append(TRAINING_KEY)

    # sort eval ones by idx, leave training last
    def idx_of(key: str) -> int:
        m = KEY_PATTERN.match(key)
        return int(m.group(1)) if m else 10 ** 6

    return sorted(keys, key=idx_of)


def fetch_full_series(run: Run, key: str) -> List[float]:
    """Fetch every recorded value for a single key via scan_history."""
    vals: List[float] = []
    for row in run.scan_history(keys=[key], page_size=10000):
        v = row.get(key)
        if v is not None:
            vals.append(v)
    return vals


def store_array(arr: List[float], path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with path.open("w") as f:
            json.dump(arr, f)
    else:
        np.savez_compressed(path.with_suffix('.npz'), data=np.asarray(arr, dtype=np.float32))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    args = cli()
    api = wandb.Api()
    base_workspace = Path(__file__).resolve().parent.parent
    ext = 'json' if args.format == 'json' else 'npz'

    for run in api.runs(args.project):
        if not want(run, args):
            continue

        cfg = run.config
        algo = cfg.get("alg_name")
        cl_method = cfg.get("cl_method", "UNKNOWN_CL")

        # Temporary fallback:
        if 'EWC' in run.name:
            cl_method = 'EWC'
        elif 'MAS' in run.name:
            cl_method = 'MAS'
        elif cl_method is None:
            cl_method = "FT"

        strategy = cfg.get("strategy")
        seq_len = cfg.get("seq_length")
        seed = max(cfg.get("seed", 1), 1)

        # find eval keys as W&B actually logged them
        eval_keys = discover_eval_keys(run)
        if not eval_keys:
            print(f"[warn] {run.name} has no Evaluation/ keys")
            continue

        out_base = (base_workspace / args.output / algo / cl_method /
                    f"{strategy}_{seq_len}" / f"seed_{seed}")

        # iterate keys, skipping existing files unless overwrite
        for key in discover_eval_keys(run):
            # choose filename
            if key == TRAINING_KEY:
                filename = f"training_reward.{ext}"
            else:
                idx, name = KEY_PATTERN.match(key).groups()
                filename = f"{idx}_{name}_reward.{ext}"

            out = out_base / filename
            if out.exists() and not args.overwrite:
                print(f"→ {out} exists, skip")
                continue

            series = fetch_full_series(run, key)
            if not series:
                print(f"→ {out} no data, skip")
                continue

            print(f"→ writing {out}")
            store_array(series, out, args.format)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
