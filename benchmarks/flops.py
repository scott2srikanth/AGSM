import torch


@torch.inference_mode()
def profile_counted_flops(model, tokens):
    """Count operations covered by PyTorch Profiler formulas."""
    model.eval()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
    ) as profile:
        model(tokens)
    operations = {row.key: row.flops for row in profile.key_averages() if row.flops}
    return {
        "profiler_counted_flops": sum(operations.values()),
        "profiled_operations": operations,
    }
