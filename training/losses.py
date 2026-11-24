from torch.nn import functional as F


def compute_losses(output, targets, settings, task="retrieval"):
    valid = targets.ne(-100)
    if task == "regression":
        prediction = output.logits.squeeze(-1)
        safe_targets = targets.masked_fill(~valid, 0)
        task_loss = (
            (prediction - safe_targets).square() * valid
        ).sum() / valid.sum().clamp_min(1)
    else:
        task_loss = F.cross_entropy(
            output.logits.flatten(0, 1), targets.flatten(), reduction="sum"
        ) / valid.sum().clamp_min(1)
    zero = task_loss.new_zeros(())
    diagnostics = output.diagnostics
    compute = diagnostics["expected_steps"].mean() if diagnostics else zero
    write = diagnostics["write_strength"].mean() if diagnostics else zero
    stability = zero
    if diagnostics:
        stability = (
            (diagnostics["fast_norm"] - settings.get("fast_threshold", 8))
            .relu()
            .square()
            + (diagnostics["slow_norm"] - settings.get("slow_threshold", 8))
            .relu()
            .square()
        ).mean()
    total = (
        task_loss
        + settings.get("compute_weight", 0) * compute
        + settings.get("write_weight", 0) * write
        + settings.get("stability_weight", 0) * stability
    )
    return {
        "total": total,
        "task": task_loss,
        "compute": compute,
        "write": write,
        "stability": stability,
    }
