import argparse
import json
from pathlib import Path

import torch

from agsm.config import merge
from tracking.report import compare_runs

from .run_experiment import run


def main():
    parser = argparse.ArgumentParser(
        description="Train the selected models in a saved budget plan"
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    directories = []
    for match in plan["matches"]:
        config = match["config"]
        name = config["model"]["name"]
        for seed in args.seeds or [config["training"]["seed"]]:
            directory = args.output / f"{name}-{match['budget']}-seed{seed}"
            run(merge(config, {"training": {"seed": seed}}), directory)
            directories.append(directory)
    compare_runs(directories, args.output / "budget_comparison.csv")


if __name__ == "__main__":
    main()
