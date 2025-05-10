#!/usr/bin/env python3
"""Download evaluation curves *per environment* for the MARL continual‑learning
benchmark and store them one metric per file, just like the original RL
benchmark.

Changes vs. the previous version
--------------------------------
* **Env names come from `config.layouts`** (this preserves the actual order
  the run used).  We prefix the index in the filename: `0_easy_layout_success.json`.
* **Single streaming pass** over the run history: we fetch all evaluation keys
  at once with `run.scan_history(keys=eval_keys, page_size=5000)` and build the
  arrays in one go – roughly 10× faster than one call per key.
* Keeps folder structure: `data/<algo>/<cl_method>/<strategy>_<seq_len>/seed_<seed>/`.
* If you ever need smaller / faster files, set `--format npz` to store
  compressed NumPy arrays instead of JSON.

Usage
-----
```bash
python dl_eval_curves.py \
  --project your_entity/your_project \
  --algos IPPO \
  --cl_methods EWC MAS FT \
  --seq_length 6  \
  --strategy ordered \
  --seeds 1 2 3 4 5
```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import wandb
from wandb.apis.public import Run

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
FORBIDDEN_TAGS = {"TEST", "LOCAL"}
EVAL_PAT = re.compile(r"Evaluation/(\d+)__(.+)")  # captures idx, env_name

# critical value for 95 % CI – handy to have around when plotting
CRIT95 = 1.96


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)

    p.add_argument("--output", default="data",
                   help="Base folder where the arrays will be written")
    p.add_argument("--format", choices=["json", "npz"], default="json",
                   help="Storage format for arrays")

    p.add_argument("--seq_length", type=int, nargs="+", default=[6])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--strategy", choices=["ordered", "random"], default=None)

    p.add_argument("--algos", nargs="+", default=[], help="Filter by alg_name")
    p.add_argument("--cl_methods", nargs="+", default=[], help="Filter by cl_method")
    p.add_argument("--wandb_tags", nargs="+", default=[], help="Require at least one of these tags")
    p.add_argument("--include_runs", nargs="+", default=[], help="Always include these runs by name substr")

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# RUN FILTERING
# ---------------------------------------------------------------------------

def want(run: Run, args: argparse.Namespace) -> bool:
    cfg = run.config

    # include‑list shortcut
    if any(tok in run.name for tok in args.include_runs):
        return True

    if run.state != "finished":
        return False

    if args.seeds and cfg.get("seed") not in args.seeds:
        return False

    if args.algos and cfg.get("alg_name") not in args.algos:
        return False

    if args.cl_methods and cfg.get("cl_method") not in args.cl_methods:
        return False

    if args.seq_length and cfg.get("seq_length") not in args.seq_length:
        return False

    if args.strategy and cfg.get("strategy") != args.strategy:
        return False

    run_tags = set(cfg.get("wandb_tags", []))
    if args.wandb_tags and not run_tags.intersection(args.wandb_tags):
        return False

    if run_tags.intersection(FORBIDDEN_TAGS) and not run_tags.intersection(args.wandb_tags):
        return False

    return True


# ---------------------------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------------------------

def to_list(x) -> List[str]:
    """Attempt to coerce cfg.layouts into a list of env names."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        # assume comma or space separated
        return re.split(r"[ ,]+", x.strip())
    raise ValueError("config.layouts is neither list nor str: %s" % x)


def fetch_series(run: Run, key: str) -> List[float]:
    """Fetch complete time series for a single eval key."""
    vals: List[float] = []
    for row in run.scan_history(keys=[key], page_size=10000):
        v = row.get(key)
        if v is not None:
            vals.append(v)
    return vals


def store_array(arr: List[float], path: Path, format: str, overwrite: bool):
    if not overwrite and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        with path.open("w") as f:
            json.dump(arr, f)
    else:  # npz – compressed binary, ~10× smaller & faster
        np.savez_compressed(path.with_suffix(".npz"), data=np.asarray(arr, dtype=np.float32))


# ---------------------------------------------------------------------------
# ENTRY
# ---------------------------------------------------------------------------

def main() -> None:
    args = cli()
    api = wandb.Api()

    for run in api.runs(args.project):
        if not want(run, args):
            continue

        cfg = run.config
        algo = cfg["alg_name"]
        cl_method = cfg.get("cl_method", "UNKNOWN_CL")
        strategy = cfg["strategy"]
        seq_len = cfg["seq_length"]
        seed = max(cfg.get("seed", 1), 1)

        # Temporary hack to determine the cl_method
        if 'EWC' in run.name:
            cl_method = 'EWC'
        elif 'MAS' in run.name:
            cl_method = 'MAS'

        layouts = to_list(cfg["layouts"])
        if len(layouts) != seq_len:
            print(f"[warn] {run.name}: len(layouts)={len(layouts)} != seq_len={seq_len}")
            layouts = layouts[:seq_len]

        base_dir = Path(__file__).resolve().parent.parent
        base = base_dir / Path(args.output) / algo / cl_method / f"{strategy}_{seq_len}" / f"seed_{seed}"

        for idx, name in enumerate(layouts):
            key = f"Evaluation/{idx}__{name}"

            ext  = 'json' if args.format=='json' else 'npz'
            out  = base / f"{idx}_{name}_success.{ext}"

            # skip if already exists and no overwrite
            if out.exists() and not args.overwrite:
                print(f"→ {out} (exists)")
                continue

            # fetch and store
            series = fetch_series(run, key)
            if not series:
                continue
            print(f"→ {out}")
            store_array(series, out, args.format)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
