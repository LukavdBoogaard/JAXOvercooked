import argparse
import os
from pathlib import Path

import wandb
from wandb.apis.public import Run

FORBIDDEN_TAGS = ['TEST', 'LOCAL']
TAG_TO_FOLDER = {}


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if suitable_run(run, args):
            store_data(run, args)


def suitable_run(run, args: argparse.Namespace) -> bool:
    try:
        # Check whether the run is in the list of runs to include by exception
        if any(logs in run.name for logs in args.include_runs):
            return True
        # Check whether the provided method corresponds to the run
        config = run.config
        # Check whether the wandb tags are suitable
        if args.wandb_tags:
            if 'wandb_tags' not in config:
                return False
            tags = config['wandb_tags']
            # Check whether the run includes one of the provided tags
            if args.wandb_tags and not any(tag in tags for tag in args.wandb_tags):
                return False
            # Check whether the run includes one of the forbidden tags which is not in the provided tags
            if any(tag in tags for tag in FORBIDDEN_TAGS) and not any(tag in tags for tag in args.wandb_tags):
                return False
        if args.algos and config['alg_name'].lower() not in args.algos.lower():
            return False

        # TODO Enable when all methods are run with cl_method
        # if args.cl_method and config['cl_method'] not in args.cl_method:
        #     return False

        # Check whether the provided sequence length corresponds to the run
        if args.seq_length and config['seq_length'] not in args.seq_length:
            return False
        # Check whether the run corresponds to one of the provided seeds
        if args.seeds and config['seed'] not in args.seeds:
            return False
        # Check whether the run corresponds to one of the provided seeds
        if args.strategy and config['strategy'] != args.strategy:
            return False
        # Check whether the run corresponds to one of the provided levels
        # if run.state not in ["finished", "crashed", 'running']:
        if run.state not in ["finished"]:
            return False
        # All filters have been passed
        return True
    except Exception as e:
        print(f"Failed to check suitability for run: {run.name}", e)
        return False


def store_data(run: Run, args: argparse.Namespace) -> None:
    config = run.config
    # level = config['level']
    cl_method = config['cl_method'] if 'cl_method' in config else 'UNKNOWN_CL'

    # Temporary hack to determine the cl_method
    if 'EWC' in run.name:
        cl_method = 'EWC'
    elif 'MAS' in run.name:
        cl_method = 'MAS'

    seq_length = config['seq_length']
    seed = config['seed']
    strategy = config['strategy']
    algo = config['alg_name']
    # tag = config['wandb_tags'][0]

    # Construct folder path for each configuration
    # folder_path = os.path.join(args.output, f"{TAG_TO_FOLDER[tag]}", algo, cl_method, f"{strategy}_{seq_length}")
    base_dir = Path(__file__).resolve().parent.parent
    folder_path = os.path.join(base_dir, args.output, algo, cl_method, f"{strategy}_{seq_length}")
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists

    # Filename based on metric
    file_name = f"seed_{seed}.csv"
    file_path = os.path.join(folder_path, file_name)

    # If the file already exists and we don't want to overwrite, skip
    if not args.overwrite and os.path.exists(file_path):
        print(f"Skipping already existing: {file_path}")
        return

    # Attempt to retrieve and save the data
    try:
        df = run.history()
        df.to_csv(file_path, index=False)
        print(f"Successfully stored run: {run.name} to {file_path}")
    except Exception as e:
        print(f"Error downloading data for run {run.name} to {file_path}", e)


def common_dl_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length", type=int, nargs='+', default=[6],
                        help="Sequence length(s) of the run(s) to download")
    parser.add_argument("--levels", type=int, nargs='+', default=[1, 2, 3], help="Level(s) of the run(s) to download")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help="Seed(s) of the run(s) to download")
    parser.add_argument("--strategy", type=str, default=[], choices=["ordered", "random"],
                        help="Sequence generation strategy")
    parser.add_argument("--algos", type=str, nargs='+', default=[], help="MARL algorithms to download")
    parser.add_argument("--cl_methods", type=str, nargs='+', default=[], help="Continual learning methods to download")
    parser.add_argument("--output", type=str, default='data', help="Base output directory to store the data")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=[], help="WandB tags to filter runs")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing files")
    parser.add_argument("--include_runs", type=str, nargs="+", default=[],
                        help="List of runs that shouldn't be filtered out")
    return parser


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())
