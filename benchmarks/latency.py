import math
import platform
import statistics
import time

import torch

from .memory import peak_process_rss, state_bytes


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def measure(model, tokens, warmup=2, repeats=7):
    if repeats < 1 or warmup < 0:
        raise ValueError("Require at least one timed repeat and nonnegative warmup")
    model.eval()
    device = tokens.device
    sync(device)
    began = time.perf_counter()
    for _ in range(warmup):
        model(tokens)
    sync(device)
    warmup_seconds = time.perf_counter() - began
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(repeats):
        sync(device)
        began = time.perf_counter()
        output = model(tokens)
        sync(device)
        durations.append(time.perf_counter() - began)
        state_size = state_bytes(output.state)
        depth = output.diagnostics.get("executed_steps")
        depth = float(depth.float().mean()) if depth is not None else None
        del output
    median = statistics.median(durations)
    return {
        "median_ms": 1000 * median,
        "p95_ms": 1000 * sorted(durations)[math.ceil(0.95 * repeats) - 1],
        "tokens_per_second": tokens.numel() / median,
        "warmup_seconds": warmup_seconds,
        "timings_ms": [seconds * 1000 for seconds in durations],
        "batch_size": tokens.shape[0],
        "sequence_length": tokens.shape[1],
        "persistent_state_bytes": state_size,
        "peak_process_rss_bytes": peak_process_rss(),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "executed_steps_mean": depth,
        "temporary_activation_bytes": None,
        "energy_per_1000_tokens": None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "platform": platform.platform(),
        "processor": platform.machine(),
        "torch": str(torch.__version__),
        "threads": torch.get_num_threads(),
    }
