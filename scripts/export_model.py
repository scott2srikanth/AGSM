import argparse
from pathlib import Path

import torch
from torch import nn

from agsm import AGSM, RecurrentState
from agsm.config import build_model
from training.checkpoint import load_checkpoint


class ExportStep(nn.Module):
    """Single-token tensor-only ABI with dense soft refinement."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token, fast, slow, valid):
        encoded = self.model.encoder(token)
        logits, state, _ = self.model.step(
            encoded, RecurrentState(fast, slow), valid, compact=False
        )
        return logits, state.fast, state.slow


def main():
    parser = argparse.ArgumentParser(
        description="Export AGSM one-token recurrent inference"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--format", choices=["torchscript", "onnx"], default="torchscript"
    )
    parser.add_argument("--int8-linear", action="store_true")
    args = parser.parse_args()
    if args.int8_linear and args.format != "torchscript":
        parser.error(
            "Dynamic INT8 linear layers are supported only for TorchScript export"
        )
    checkpoint = load_checkpoint(args.checkpoint)
    model = build_model(checkpoint["config"])
    if not isinstance(model, AGSM):
        parser.error("Export requires an AGSM checkpoint")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if args.int8_linear:
        if torch.backends.quantized.engine == "none":
            engines = torch.backends.quantized.supported_engines
            if "qnnpack" not in engines:
                parser.error(
                    "This PyTorch build has no enabled INT8 quantization engine"
                )
            torch.backends.quantized.engine = "qnnpack"
        model = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
    module = ExportStep(model).eval()
    example = (
        torch.zeros(2, dtype=torch.long),
        model.initialize_state(2).fast,
        model.initialize_state(2).slow,
        torch.ones(2, dtype=torch.bool),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        if args.format == "torchscript":
            exported = torch.jit.trace(module, example)
            exported.save(str(output))
        else:
            inputs, outputs = (
                ["token", "fast", "slow", "valid"],
                ["logits", "next_fast", "next_slow"],
            )
            torch.onnx.export(
                module,
                example,
                str(output),
                opset_version=17,
                dynamo=False,
                input_names=inputs,
                output_names=outputs,
                dynamic_axes={name: {0: "batch"} for name in inputs + outputs},
            )


if __name__ == "__main__":
    main()
