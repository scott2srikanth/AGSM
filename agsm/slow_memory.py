import torch
from torch import Tensor, nn


class SlowContextualMemory(nn.Module):
    """GRU context with an additional controller-driven interpolation gate."""

    def __init__(self, model_dim: int, value_dim: int, slow_dim: int):
        super().__init__()
        self.slow_dim = slow_dim
        self.cell = nn.GRUCell(model_dim + value_dim, slow_dim)

    def initial_state(self, batch_size: int, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch_size, self.slow_dim, device=device, dtype=dtype)

    def reset_state(self, state: Tensor) -> Tensor:
        return torch.zeros_like(state)

    def forward(
        self, encoded: Tensor, retrieval: Tensor, state: Tensor, gate: Tensor
    ) -> Tensor:
        candidate = self.cell(torch.cat((encoded, retrieval), dim=-1), state)
        return torch.lerp(state, candidate.to(state.dtype), gate.to(state.dtype))
