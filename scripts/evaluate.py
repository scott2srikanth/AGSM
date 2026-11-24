import argparse
from pathlib import Path

from agsm.config import build_model
from evaluation.evaluate import evaluate
from tracking.artifact_manager import write_json
from training.checkpoint import load_checkpoint
from training.trainer import select_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    device = select_device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    result = evaluate(model, config, device, args.split)
    result["metadata"] = checkpoint["metadata"]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)


if __name__ == "__main__":
    main()
