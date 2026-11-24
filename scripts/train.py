import argparse

import torch

from agsm.config import load_config, validate_config
from training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train AGSM or a local baseline")
    parser.add_argument("--config", default="configs/training/default.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    validate_config(config)
    metadata, history = train(config, args.output, args.resume)
    print(
        f"{metadata['experiment_id']}: {len(history)} recorded epochs in {args.output}"
    )


if __name__ == "__main__":
    main()
