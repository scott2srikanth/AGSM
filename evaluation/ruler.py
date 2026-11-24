import argparse
import json
from pathlib import Path

import torch

from agsm import AGSM
from agsm.config import build_model
from data.ruler import read_ruler
from training.checkpoint import load_checkpoint
from training.trainer import select_device


@torch.inference_mode()
def predict(model, prompt, limit, device, chunk_size):
    prompt = list(prompt.encode("utf-8"))
    if not prompt:
        raise ValueError("RULER prompt cannot be empty")
    state = None
    for offset in range(0, len(prompt), chunk_size):
        tokens = torch.tensor([prompt[offset : offset + chunk_size]], device=device)
        output = model.process_chunk(tokens, state)
        state = output.state
    generated = []
    for _ in range(limit):
        token = output.logits[:, -1].argmax(-1)
        generated.append(token.item())
        output = model.process_chunk(token[:, None], state)
        state = output.state
    return bytes(generated).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for upstream RULER scoring"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--max-new-bytes", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.chunk_size, args.max_new_bytes) < 1:
        parser.error("Chunk size and generation budget must be positive")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if (
        config["data"]["name"] != "wikitext103"
        or config["model"].get("task") != "language_model"
    ):
        parser.error("Requires a UTF-8 byte language-model checkpoint")
    if (
        config["model"].get("output_dim", 256) != 256
        or config["model"]["vocab_size"] != 256
    ):
        parser.error(
            "Byte language model must have exactly 256 input and output symbols"
        )
    device = select_device(args.device)
    model = build_model(config).to(device).eval()
    if not isinstance(model, AGSM):
        parser.error("This prediction generator requires an AGSM streaming checkpoint")
    model.load_state_dict(checkpoint["model"])
    with Path(args.output).open("x", encoding="utf-8") as destination:
        for example in read_ruler(args.dataset):
            prediction = predict(
                model, example["input"], args.max_new_bytes, device, args.chunk_size
            )
            destination.write(json.dumps({**example, "pred": prediction}) + "\n")


if __name__ == "__main__":
    main()
