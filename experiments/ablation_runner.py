import argparse
from pathlib import Path

import torch

from agsm.config import load_config, merge
from tracking.report import compare_runs

from .run_experiment import run


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled YAML variants and collect results"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    variants = config.pop("sweep")
    selected = args.variants or variants
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    runs = []
    for name in selected:
        for seed in args.seeds:
            directory = root / f"{name}-seed{seed}"
            resolved = merge(
                merge(config, variants[name]), {"training": {"seed": seed}}
            )
            run(resolved, directory)
            runs.append(directory)
    compare_runs(runs, root / "ablation_comparison.csv")


if __name__ == "__main__":
    main()
