import argparse
from pathlib import Path

import torch

from agsm.config import build_model, load_config
from benchmarks.flops import profile_counted_flops
from benchmarks.latency import measure
from tracking.artifact_manager import write_json
from training.checkpoint import load_checkpoint
from training.seed import seed_everything
from training.trainer import select_device


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a model on deterministic synthetic inputs"
    )
    parser.add_argument("--config", default="configs/benchmarks/benchmark.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    options = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    config = checkpoint["config"] if checkpoint else options
    seed_everything(config["training"]["seed"])
    model = build_model(config).to(select_device(args.device))
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    settings = options["benchmark"]
    rows = []
    for length in args.lengths or settings["lengths"]:
        tokens = torch.randint(
            config["model"]["vocab_size"],
            (settings["batch_size"], length),
            device=next(model.parameters()).device,
        )
        result = measure(model, tokens, settings["warmup"], settings["repeats"])
        if args.profile:
            result.update(profile_counted_flops(model, tokens))
        rows.append(result)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, {"model": config["model"], "measurements": rows})


if __name__ == "__main__":
    main()
