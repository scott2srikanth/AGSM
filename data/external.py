import csv
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset


class WikiText103(Dataset):
    """Byte-level next-token blocks from a locally staged WikiText-103 raw split."""

    def __init__(self, path, split, length, limit, revision):
        path = Path(path)
        self.content = path.read_bytes()
        self.length = length
        self.size = min(limit, (len(self.content) - 1) // length)
        if self.size < 1:
            raise ValueError(
                f"WikiText-103 split {split} is too short for the configured length"
            )
        self.metadata = {
            "name": "wikitext103",
            "version": revision,
            "split": split,
            "sequence_length": length,
            "vocabulary_version": "utf8-byte-v1",
            "examples": self.size,
            "sha256": hashlib.sha256(self.content).hexdigest(),
        }

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        start = index * self.length
        tokens = torch.tensor(
            list(self.content[start : start + self.length + 1]), dtype=torch.long
        )
        return tokens[:-1], tokens[1:]


class ListOps(Dataset):
    """Read canonical LRA ListOps TSV files using a shared fixed symbol vocabulary."""

    SYMBOLS = dict(
        zip("0 1 2 3 4 5 6 7 8 9 [MAX [MIN [MED [SM ]".split(), range(1, 16))
    )

    def __init__(self, path, split, length, limit, revision):
        path = Path(path)
        self.samples = []
        self.length = length
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                expression = row["Source"].translate(
                    str.maketrans({"(": " ", ")": " ", "]": " ] "})
                )
                symbols = expression.split()
                if len(symbols) > length:
                    continue
                self.samples.append(
                    ([self.SYMBOLS[symbol] for symbol in symbols], int(row["Target"]))
                )
                if len(self.samples) == limit:
                    break
        if not self.samples:
            raise ValueError(f"No ListOps examples fit sequence length {length}")
        self.metadata = {
            "name": "listops",
            "version": revision,
            "split": split,
            "sequence_length": length,
            "vocabulary_version": "listops-symbols-v1",
            "examples": len(self.samples),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        symbols, label = self.samples[index]
        # Left padding allows the task head to be supervised at the final real symbol.
        tokens = torch.zeros(self.length, dtype=torch.long)
        targets = torch.full_like(tokens, -100)
        tokens[-len(symbols) :] = torch.tensor(symbols)
        targets[-1] = label
        return tokens, targets
