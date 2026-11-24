from pathlib import Path

import torch

from agsm.config import load_config
from data.data_utils import make_dataset
from training.checkpoint import load_checkpoint
from training.trainer import train


def test_checkpoint_recovery_and_deterministic_resume(tmp_path):
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/training/smoke.yaml"
    )
    directory = tmp_path / "run"
    metadata, history = train(config, directory)
    original = load_checkpoint(directory / "best.pt")
    (directory / "epoch-000002.pt").write_bytes(b"invalid checkpoint")
    resumed_metadata, resumed_history = train(config, directory, resume=True)
    resumed = load_checkpoint(directory / "best.pt")
    assert metadata == resumed_metadata
    assert len(history) == len(resumed_history) == 2
    assert (
        resumed_history[1]["validation/loss_task"] == history[1]["validation/loss_task"]
    )
    for key in original["model"]:
        torch.testing.assert_close(
            original["model"][key], resumed["model"][key], rtol=0, atol=0
        )


def test_synthetic_split_reproducibility():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/training/smoke.yaml"
    )
    train_set = make_dataset(config, "train")
    validation = make_dataset(config, "validation")
    torch.testing.assert_close(train_set[0][0], train_set[0][0])
    assert not torch.equal(train_set[0][0], validation[0][0])
