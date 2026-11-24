import os
import random
import tempfile
from pathlib import Path

import torch


def save_checkpoint(path, payload):
    """Atomically replace a checkpoint within the same filesystem."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        try:
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(temporary.name, path)
        finally:
            Path(temporary.name).unlink(missing_ok=True)


def load_checkpoint(path):
    """Load tensors and primitive metadata from a local research checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=True)


def rng_state(device):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng_state(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)
