import argparse
from pathlib import Path

import torch

from agsm.config import build_model, load_config, merge
from benchmarks.flops import profile_counted_flops
from tracking.artifact_manager import write_json
from training.trainer import select_device


def main():
    parser = argparse.ArgumentParser(
        description="Find nearest candidate width for each budget"
    )
    parser.add_argument("--config", default="configs/experiments/capacity.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models", nargs="+", default=["agsm_v2", "gru", "transformer"]
    )
    parser.add_argument("--flop-budgets", nargs="+", type=int)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.config)
    torch.manual_seed(config["training"]["seed"])
    if min(args.length, args.batch_size) < 1:
        parser.error("Length and batch size must be positive")
    device = select_device(args.device)
    tokens = torch.randint(
        config["model"]["vocab_size"], (args.batch_size, args.length), device=device
    )
    rows = []
    field = "profiler_counted_flops" if args.flop_budgets else "parameters"
    for name in args.models:
        candidates = []
        divisor = {
            "transformer": 4,
            "rwkv7": 64,
            "mamba2": 32,
            "gated_deltanet": 32,
        }.get(name, 1)
        for width in config["candidates"]:
            if width % divisor:
                continue
            variant = merge(
                config,
                {
                    "model": {
                        "name": name,
                        "model_dim": width,
                        "slow_state_dim": width,
                        "controller_dim": width,
                        "refinement_hidden_dim": width * 2,
                        "key_dim": width // 2,
                        "value_dim": width // 2,
                    }
                },
            )
            model = build_model(variant)
            row = {
                "config": variant,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            if args.flop_budgets:
                row.update(profile_counted_flops(model.to(device), tokens))
            candidates.append(row)
        if not candidates:
            raise ValueError(f"No compatible candidate widths for {name}")
        for budget in args.flop_budgets or config["budgets"]:
            match = min(candidates, key=lambda row: abs(row[field] - budget))
            rows.append({**match, "budget": budget, "deviation": match[field] - budget})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        destination,
        {
            "budget_type": field,
            "sequence_length": args.length,
            "batch_size": args.batch_size,
            "device": str(device),
            "torch": str(torch.__version__),
            "matches": rows,
        },
    )


if __name__ == "__main__":
    main()
