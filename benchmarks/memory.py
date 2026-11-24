import resource
import sys

import torch


def state_bytes(state):
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, (tuple, list)):
        parts = [state_bytes(part) for part in state]
        return sum(parts) if all(part is not None for part in parts) else None
    return getattr(state, "nbytes", None)


def peak_process_rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024
