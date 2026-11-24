from __future__ import annotations

import math
from pathlib import Path

import yaml

from .model import AGSM


def merge(base: dict, overrides: dict) -> dict:
    """Merge mappings recursively, replacing scalar and list values."""
    result = dict(base)
    for key, value in overrides.items():
        previous = result.get(key)
        result[key] = (
            merge(previous, value)
            if isinstance(previous, dict) and isinstance(value, dict)
            else value
        )
    return result


def load_config(path: str | Path, ancestors: tuple[Path, ...] = ()) -> dict:
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError(f"Circular configuration inheritance at {path}")
    with path.open() as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    parents = config.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    result = {}
    for parent in parents:
        result = merge(result, load_config(path.parent / parent, (*ancestors, path)))
    return merge(result, config)


def validate_config(config: dict) -> None:
    model = config["model"]
    if model.get("name") not in {
        "agsm_v2",
        "gru",
        "transformer",
        "mamba2",
        "rwkv7",
        "gated_deltanet",
    }:
        raise ValueError("Unknown or unconfigured model implementation")
    for field in (
        "vocab_size",
        "model_dim",
        "key_dim",
        "value_dim",
        "slow_state_dim",
        "controller_dim",
        "refinement_hidden_dim",
    ):
        if type(model[field]) is not int or model[field] <= 0:
            raise ValueError(f"model.{field} must be a positive integer")
    if model["vocab_size"] < 8:
        raise ValueError("vocab_size must reserve room for task markers and payload")
    if (
        type(model["max_refinement_steps"]) is not int
        or model["max_refinement_steps"] < 0
    ):
        raise ValueError("max_refinement_steps must be a nonnegative integer")
    memory = config.get("memory", {})
    if memory.get("type", "dual") not in {"dual", "fast", "slow"}:
        raise ValueError("Unknown memory type")
    if memory.get("read_before_write", True) is not True:
        raise ValueError("AGSM requires read-before-write")
    if memory.get("update_precision", "float32") not in {
        "float32",
        "float64",
        "float16",
        "bfloat16",
    }:
        raise ValueError("Unsupported memory precision")
    if not 0 < memory.get("norm_limit", 16) < math.inf:
        raise ValueError("memory.norm_limit must be finite and positive")
    threshold = config.get("evaluation", {}).get("inference_threshold", 0.99)
    if not 0 < threshold <= 1:
        raise ValueError("Inference threshold must be in (0, 1]")
    if model.get("task", "retrieval") not in {
        "classification",
        "retrieval",
        "regression",
        "language_model",
    }:
        raise ValueError("Unknown task head")
    training = config.get("training", {})
    for field in ("epochs", "batch_size", "patience"):
        if field in training and (
            type(training[field]) is not int or training[field] <= 0
        ):
            raise ValueError(f"training.{field} must be a positive integer")
    if training.get("optimizer", "adamw") != "adamw":
        raise ValueError("Only AdamW is supported")
    for name, value in config.get("loss", {}).items():
        if not isinstance(value, (int, float)) or not 0 <= value < math.inf:
            raise ValueError(f"loss.{name} must be finite and nonnegative")
    checkpoint = config.get("checkpoint", {})
    interval = checkpoint.get("save_every_epochs", 1)
    if type(interval) is not int or interval < 1:
        raise ValueError("checkpoint.save_every_epochs must be a positive integer")
    for name in ("learning_rate", "gradient_clip", "weight_decay"):
        if name in training:
            value = training[name]
            minimum = 0 if name == "weight_decay" else 1e-15
            if not isinstance(value, (int, float)) or not minimum <= value < math.inf:
                raise ValueError(f"training.{name} is outside its allowed range")
    data = config.get("data", {})
    if data:
        from data.data_utils import TASKS

        if data["name"] not in {*TASKS, "listops", "wikitext103"}:
            raise ValueError(f"Unknown dataset {data['name']}")
        for name in ("train_examples", "validation_examples", "test_examples"):
            if type(data[name]) is not int or data[name] < 1:
                raise ValueError(f"data.{name} must be a positive integer")
        if type(data["sequence_length"]) is not int or data["sequence_length"] < 6:
            raise ValueError("data.sequence_length must be an integer of at least six")
        if data["name"] == "wikitext103" and (
            model.get("task") != "language_model"
            or model["vocab_size"] != 256
            or model.get("output_dim", 256) != 256
        ):
            raise ValueError(
                "WikiText-103 byte modeling requires 256 input and output symbols"
            )
    if config.get("tracking", {}).get("backend", "json") not in {"json", "wandb"}:
        raise ValueError("Unsupported tracking backend")


def build_model(config: dict):
    validate_config(config)
    options = dict(config["model"])
    name = options.pop("name")
    if name == "gru":
        from baselines.gru import GRUBaseline

        return GRUBaseline(**options)
    if name == "transformer":
        from baselines.transformer import TransformerBaseline

        return TransformerBaseline(**options)
    if name == "mamba2":
        from baselines.mamba_adapter import Mamba2Adapter

        return Mamba2Adapter(**options)
    if name == "rwkv7":
        from baselines.rwkv_adapter import RWKV7Adapter

        return RWKV7Adapter(**options)
    if name == "gated_deltanet":
        from baselines.gated_deltanet_adapter import GatedDeltaNetAdapter

        return GatedDeltaNetAdapter(**options)
    return AGSM(
        **options,
        memory=config.get("memory"),
        controller=config.get("controller"),
        inference_threshold=config.get("evaluation", {}).get(
            "inference_threshold", 0.99
        ),
    )
