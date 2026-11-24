import torch
from torch.utils.data import Dataset

from . import delayed_recall, mqar, sequence_copy, variable_lag
from .external import ListOps, WikiText103

TASKS = {
    "delayed_recall": delayed_recall.generate,
    "sequence_copy": sequence_copy.generate,
    "variable_lag": variable_lag.generate,
    "mqar": mqar.generate,
}
SPLITS = {"train": 0, "validation": 1, "test": 2}


class SyntheticDataset(Dataset):
    """Index-addressable synthetic sequence split with documented RNG streams."""

    def __init__(self, name, split, size, length, vocab_size, seed):
        if name not in TASKS or split not in SPLITS:
            raise ValueError(f"Unknown synthetic task or split: {name}/{split}")
        if size < 1 or size >= 2**40 or length < 6 or vocab_size < 8:
            raise ValueError(
                "Require 1 <= size < 2**40, length >= 6 and vocabulary >= 8"
            )
        self.size, self.length, self.vocab_size = size, length, vocab_size
        # PyTorch CPU generators can alias seeds that share the same low 32 bits.
        # Allocate interleaved split IDs and mix all input bits before initialization.
        self.seed = seed
        self.split = SPLITS[split]
        self.generate = TASKS[name]
        self.metadata = {
            "name": name,
            "version": "synthetic-v1",
            "split": split,
            "sequence_length": length,
            "vocabulary_version": "markers-v1",
            "vocab_size": vocab_size,
            "seed": seed,
            "examples": size,
        }

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        import hashlib

        identity = f"{self.seed}:{self.split}:{index}".encode()
        seed = int.from_bytes(
            hashlib.blake2b(identity, digest_size=8).digest(), "little"
        )
        rng = torch.Generator().manual_seed(seed)
        tokens = torch.randint(3, self.vocab_size, (self.length,), generator=rng)
        return self.generate(tokens, self.vocab_size, rng)


def make_dataset(config, split):
    spec = config["data"]
    if spec["name"] in {"wikitext103", "listops"}:
        dataset = WikiText103 if spec["name"] == "wikitext103" else ListOps
        return dataset(
            spec["paths"][split],
            split,
            spec["sequence_length"],
            spec[f"{split}_examples"],
            spec["revision"],
        )
    return SyntheticDataset(
        spec["name"],
        split,
        spec[f"{split}_examples"],
        spec["sequence_length"],
        config["model"]["vocab_size"],
        config["training"]["seed"],
    )
