from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.functional import normalize


class FastAssociativeMemory(nn.Module):
    """Fixed-size delta-rule memory with explicit read and write operations."""

    def __init__(
        self,
        value_dim: int,
        key_dim: int,
        precision: str = "float32",
        norm_limit: float | None = None,
        normalize_keys: bool = True,
    ):
        super().__init__()
        self.value_dim = value_dim
        self.key_dim = key_dim
        self.dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]
        self.norm_limit = norm_limit
        self.normalize_keys = normalize_keys

    def initial_state(self, batch_size: int, device=None) -> Tensor:
        return torch.zeros(
            batch_size, self.value_dim, self.key_dim, device=device, dtype=self.dtype
        )

    def reset_state(self, state: Tensor) -> Tensor:
        return torch.zeros_like(state)

    def address(self, vector: Tensor) -> Tensor:
        compute_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
        vector = vector.to(compute_dtype)
        return normalize(vector, dim=-1, eps=1e-6) if self.normalize_keys else vector

    def read(self, state: Tensor, address: Tensor) -> Tensor:
        with torch.autocast(device_type=state.device.type, enabled=False):
            return torch.bmm(state.to(address.dtype), address.unsqueeze(-1)).squeeze(-1)

    def write(
        self,
        state: Tensor,
        key: Tensor,
        residual: Tensor,
        retention: Tensor,
        strength: Tensor,
    ) -> Tensor:
        updated = retention.unsqueeze(-1) * state.to(key.dtype)
        updated = updated + (strength * residual).unsqueeze(-1) * key.unsqueeze(-2)
        if self.norm_limit is not None:
            norm = self.norm(updated)
            scale = self.norm_limit / norm.clamp_min(self.norm_limit)
            updated = updated * scale[:, None, None]
        return updated.to(self.dtype)

    @staticmethod
    def norm(state: Tensor) -> Tensor:
        dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
        return torch.linalg.vector_norm(state.to(dtype), dim=(-2, -1))
