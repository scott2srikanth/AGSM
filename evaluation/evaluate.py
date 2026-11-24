import torch
from torch.utils.data import DataLoader

from data.data_utils import make_dataset
from evaluation.metrics import MetricAccumulator
from training.losses import compute_losses


@torch.inference_mode()
def evaluate(model, config, device, split="validation", *, dataset=None):
    model.eval()
    stats = MetricAccumulator()
    if dataset is None:
        dataset = make_dataset(config, split)
    batches = DataLoader(
        dataset,
        batch_size=config.get("evaluation", {}).get(
            "batch_size", config["training"]["batch_size"]
        ),
    )
    task = config["model"].get("task", "retrieval")
    refinement_counts = {}
    for tokens, targets in batches:
        tokens, targets = tokens.to(device), targets.to(device)
        mask = tokens.ne(0) if config["data"]["name"] == "listops" else None
        output = model(tokens, mask=mask)
        losses = compute_losses(output, targets, config.get("loss", {}), task)
        stats.update(output, targets, losses, task)
        steps = output.diagnostics.get("executed_steps")
        if steps is not None:
            for depth, count in enumerate(
                steps.flatten()
                .bincount(minlength=config["model"]["max_refinement_steps"] + 1)
                .tolist()
            ):
                if count:
                    refinement_counts[depth] = refinement_counts.get(depth, 0) + count
    return {
        "dataset": dataset.metadata,
        "metrics": stats.result(),
        "refinement_counts": refinement_counts,
    }
