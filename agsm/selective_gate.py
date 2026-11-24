from typing import NamedTuple

from torch import Tensor, nn


class GateFeatures(NamedTuple):
    encoded: Tensor
    query: Tensor
    key: Tensor
    value: Tensor


class SelectiveInformationGate(nn.Module):
    """Project input and context into addressing and payload features."""

    def __init__(self, model_dim: int, slow_dim: int, key_dim: int, value_dim: int):
        super().__init__()
        self.input = nn.Linear(model_dim, model_dim)
        self.context = nn.Linear(slow_dim, model_dim, bias=False)
        self.norm = nn.LayerNorm(model_dim)
        self.query = nn.Linear(model_dim, key_dim)
        self.key = nn.Linear(model_dim, key_dim)
        self.value = nn.Linear(model_dim, value_dim)

    def forward(self, x: Tensor, slow: Tensor) -> GateFeatures:
        u = self.norm(self.input(x) + self.context(slow))
        return GateFeatures(u, self.query(u), self.key(u), self.value(u))
