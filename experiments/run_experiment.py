import argparse
from pathlib import Path

import torch

from agsm.config import build_model, load_config
from benchmarks.latency import measure
from evaluation.evaluate import evaluate
from evaluation.long_context import evaluate_lengths
from tracking.report import write_report
from training.checkpoint import load_checkpoint
from training.trainer import select_device, train


def run(config, directory, resume=False):
    directory = Path(directory)
    metadata, history = train(config, directory, resume)
    device = select_device(config["training"]["device"])
    model = build_model(config).to(device)
    model.load_state_dict(load_checkpoint(directory / "best.pt")["model"])
    test = evaluate(model, config, device, "test")
    lengths = evaluate_lengths(model, config, device)
    benchmarks = []
    for length in config["evaluation"]["sequence_lengths"]:
        generator = torch.Generator().manual_seed(config["training"]["seed"])
        tokens = torch.randint(
            config["model"]["vocab_size"], (1, length), generator=generator
        )
        benchmarks.append(measure(model, tokens.to(device)))
    write_report(
        directory,
        metadata,
        history,
        test,
        lengths,
        benchmarks,
        test["refinement_counts"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(load_config(args.config), args.output, args.resume)


if __name__ == "__main__":
    main()
