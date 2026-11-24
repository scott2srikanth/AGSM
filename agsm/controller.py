from __future__ import annotations

import torch
from torch import Tensor, nn


class JointController(nn.Module):
    """Coordinate memory operations and refinement using one shared context."""

    DEFAULTS = {
        "write": 1.0,
        "slow_update": 1.0,
        "retrieval": 1.0,
        "update_strength": 1.0,
        "retention": 1.0,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        model_dim: int,
        adaptive: dict[str, bool] | None = None,
    ):
        super().__init__()
        self.adaptive = dict(adaptive or {})
        self.context = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.operations = nn.Linear(hidden_dim, len(self.DEFAULTS))
        self.halt = nn.Linear(hidden_dim + model_dim, 1)

    def forward(self, features: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        context = self.context(features)
        probabilities = self.operations(context).float().sigmoid().split(1, dim=-1)
        gates = {
            name: probability
            if self.adaptive.get(name, True)
            else torch.full_like(probability, fallback)
            for (name, fallback), probability in zip(
                self.DEFAULTS.items(), probabilities
            )
        }
        return gates, context

    def halting_probability(self, context: Tensor, representation: Tensor) -> Tensor:
        return self.halt(torch.cat((context, representation), dim=-1)).float().sigmoid()
