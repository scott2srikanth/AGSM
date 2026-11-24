from typing import NamedTuple

import torch
from torch import Tensor, nn


class RefinementResult(NamedTuple):
    representation: Tensor
    expected: Tensor
    executed: Tensor


class AdaptiveRefinement(nn.Module):
    """Mix depth representations using a truncated conditional stopping distribution."""

    def __init__(
        self,
        model_dim: int,
        context_dim: int,
        hidden_dim: int,
        max_steps: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.initial = nn.Linear(context_dim, model_dim)
        self.delta = nn.Sequential(
            nn.Linear(model_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def update(self, z: Tensor, features: Tensor) -> Tensor:
        return self.norm(z + self.delta(torch.cat((z, features), dim=-1)))

    def forward(
        self,
        features: Tensor,
        context: Tensor,
        halt,
        threshold: float,
        adaptive: bool = True,
        compact: bool = True,
    ) -> RefinementResult:
        z = self.initial(features)
        survival = z.new_ones((z.shape[0], 1), dtype=torch.float32)
        expected = torch.zeros_like(survival)
        mixed = torch.zeros_like(z)
        executed = torch.zeros_like(survival, dtype=torch.long)
        rows = torch.arange(z.shape[0], device=z.device)
        compact = compact and not self.training and adaptive

        for depth in range(1, self.max_steps + 1):
            z = self.update(z, features)
            if depth == self.max_steps:
                mass = survival
                stopped = torch.ones_like(survival, dtype=torch.bool)
            else:
                mass = (
                    survival * halt(context, z)
                    if adaptive
                    else torch.zeros_like(survival)
                )
                stopped = (
                    (1 - survival + mass >= threshold)
                    if compact
                    else torch.zeros_like(survival, dtype=torch.bool)
                )
                mass = torch.where(stopped, survival, mass)
            weighted = z * mass.to(z.dtype)
            survival = survival - mass
            if compact:
                mixed = mixed.index_add(0, rows, weighted)
                expected = expected.index_add(0, rows, mass * depth)
                executed = executed.index_add(
                    0, rows, torch.ones_like(mass, dtype=torch.long)
                )
                active = (~stopped.squeeze(-1)).nonzero(as_tuple=True)[0]
                if active.numel() == 0:
                    break
                rows, z, features, context, survival = (
                    tensor.index_select(0, active)
                    for tensor in (rows, z, features, context, survival)
                )
            else:
                mixed = mixed + weighted
                expected = expected + mass * depth
                executed = executed + 1
        if self.max_steps == 0:
            mixed = z
        return RefinementResult(mixed, expected.squeeze(-1), executed.squeeze(-1))
