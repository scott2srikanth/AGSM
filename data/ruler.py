import json
from pathlib import Path


def read_ruler(path):
    """Read NVIDIA RULER prepare.py JSONL examples for a pinned task and length."""
    with Path(path).open(encoding="utf-8") as source:
        for row in source:
            example = json.loads(row)
            if not isinstance(example["input"], str) or not isinstance(
                example["outputs"], list
            ):
                raise ValueError("Expected NVIDIA RULER input text and output list")
            yield example
