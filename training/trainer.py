import pickle
import time
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from agsm.config import build_model, validate_config
from data.data_utils import make_dataset
from evaluation.evaluate import evaluate
from evaluation.metrics import MetricAccumulator
from tracking.artifact_manager import write_json
from tracking.experiment_logger import ExperimentLogger, git_metadata
from training.checkpoint import (
    load_checkpoint,
    restore_rng_state,
    rng_state,
    save_checkpoint,
)
from training.losses import compute_losses
from training.optimizer import make_optimizer
from training.scheduler import make_scheduler
from training.seed import seed_everything


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(model, config, device, batches, optimizer, scaler):
    settings = config["training"]
    task = config["model"].get("task", "retrieval")
    metrics = MetricAccumulator()
    model.train()
    for tokens, targets in batches:
        tokens, targets = tokens.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            mask = tokens.ne(0) if config["data"]["name"] == "listops" else None
            output = model(tokens, mask=mask)
            losses = compute_losses(output, targets, config.get("loss", {}), task)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("Nonfinite training loss")
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            settings["gradient_clip"],
            error_if_nonfinite=not scaler.is_enabled(),
        )
        scaler.step(optimizer)
        scaler.update()
        if torch.isfinite(norm):
            metrics.add("gradient_norm", norm, tokens.shape[0])
        metrics.update(output, targets, losses, task)
    return metrics.result()


def _resume_checkpoint(directory):
    paths = sorted(directory.glob("epoch-*.pt"), reverse=True)
    if not paths:
        raise FileNotFoundError(f"No epoch checkpoints in {directory}")
    for path in paths:
        try:
            state = load_checkpoint(path)
            required = {
                "epoch",
                "config",
                "model",
                "optimizer",
                "scheduler",
                "scaler",
                "rng",
                "history",
                "best_loss",
                "stale",
                "metadata",
                "best_model",
            }
            if isinstance(state, dict) and required <= state.keys():
                return state
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
            continue
    raise RuntimeError(f"No readable epoch checkpoint in {directory}")


def train(config, directory, resume=False):
    """Train or resume at the next epoch using a durable optimizer/RNG snapshot."""
    validate_config(config)
    settings = config["training"]
    device = select_device(settings["device"])
    seed_everything(settings["seed"])
    directory = Path(directory)
    if resume and not config.get("checkpoint", {}).get("resume", True):
        raise ValueError("Resume is disabled in the resolved checkpoint configuration")
    if resume:
        checkpoint = _resume_checkpoint(directory)
        if checkpoint["config"] != config:
            raise ValueError("Resume requires the same resolved configuration")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = None
    model = build_model(config).to(device)
    optimizer = make_optimizer(model, settings)
    scheduler = make_scheduler(optimizer, settings)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=settings["mixed_precision"] and device.type == "cuda"
    )
    dataset = make_dataset(config, "train")
    validation_dataset = make_dataset(config, "validation")
    generator = torch.Generator()
    batches = DataLoader(
        dataset, batch_size=settings["batch_size"], shuffle=True, generator=generator
    )
    metadata = {
        "experiment_id": uuid.uuid4().hex,
        "model_version": "0.1.0",
        "config": config,
        "git": git_metadata(),
        "dataset": dataset.metadata,
        "validation_dataset": validation_dataset.metadata,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    active_gpu = metadata["gpu"]
    history, best, stale, start_epoch = [], float("inf"), 0, 1
    best_model = None
    if checkpoint is not None:
        original = checkpoint["metadata"]
        for field in ("git", "dataset", "validation_dataset", "device", "torch"):
            if original[field] != metadata[field]:
                raise ValueError(f"Cannot resume with changed run metadata: {field}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng"], device)
        history = checkpoint["history"]
        best, stale = checkpoint["best_loss"], checkpoint["stale"]
        start_epoch = checkpoint["epoch"] + 1
        metadata = checkpoint["metadata"]
        best_model = checkpoint["best_model"]

    def save_epoch(epoch):
        save_checkpoint(
            directory / f"epoch-{epoch:06}.pt",
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(device),
                "config": config,
                "metadata": metadata,
                "history": history,
                "best_loss": best,
                "stale": stale,
                "best_model": best_model,
            },
        )

    if checkpoint is None:
        save_epoch(0)
    write_json(directory / "metadata.json", metadata)
    logger = ExperimentLogger(config.get("tracking", {}), metadata, resume)
    try:
        for epoch in range(start_epoch, settings["epochs"] + 1):
            if stale >= settings["patience"]:
                break
            began = time.perf_counter()
            generator.manual_seed(settings["seed"] + epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            training = train_epoch(model, config, device, batches, optimizer, scaler)
            validation = evaluate(model, config, device, dataset=validation_dataset)[
                "metrics"
            ]
            row = {
                "epoch": epoch,
                "gpu": active_gpu,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - began,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                **{f"train/{key}": value for key, value in training.items()},
                **{f"validation/{key}": value for key, value in validation.items()},
            }
            score = validation["loss_task"]
            if not torch.isfinite(torch.tensor(score)):
                raise FloatingPointError("Nonfinite validation loss")
            if score < best:
                best, stale = score, 0
                best_model = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale += 1
            history.append(row)
            scheduler.step()
            interval = config.get("checkpoint", {}).get("save_every_epochs", 1)
            if (
                epoch % interval == 0
                or epoch == settings["epochs"]
                or stale >= settings["patience"]
            ):
                save_epoch(epoch)
            write_json(directory / "metrics.json", history)
            logger.log(row)
        write_json(directory / "metrics.json", history)
        save_checkpoint(
            directory / "best.pt",
            {"model": best_model, "config": config, "metadata": metadata},
        )
    finally:
        logger.close()
    return metadata, history
