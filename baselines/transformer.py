import math

import torch
from torch import nn

from agsm.model import ModelOutput


class TransformerBaseline(nn.Module):
    """Two-layer causal Transformer with length-independent sinusoidal positions."""

    def __init__(self, vocab_size, model_dim, output_dim=None, dropout=0.0, **_):
        super().__init__()
        if model_dim % 4:
            raise ValueError("Transformer model_dim must be divisible by four")
        self.embedding = nn.Embedding(vocab_size, model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim,
            4,
            model_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.head = nn.Linear(model_dim, output_dim or vocab_size)

    def forward(self, tokens, state=None, mask=None):
        if state is not None:
            raise ValueError(
                "This full-context Transformer baseline has no streaming adapter"
            )
        x = self.embedding(tokens)
        length, width = x.shape[1:]
        indices = torch.arange(length, device=x.device)
        positions = (
            indices[None, :]
            if mask is None
            else (mask.long().cumsum(-1) - 1).clamp_min(0)
        )
        frequency = torch.arange(0, width, 2, device=x.device).float() / width
        angles = positions[..., None] * torch.exp(-math.log(10000) * frequency)
        embedding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        causal = indices[:, None] < indices[None, :]
        if mask is not None:
            # Allow a masked query to attend to itself to avoid all-masked attention.
            causal = causal | (
                ~mask[:, None, :]
                & ~torch.eye(length, dtype=torch.bool, device=x.device)
            )
            causal = causal.repeat_interleave(4, dim=0)
        hidden = self.layers(x + embedding.to(x.dtype), mask=causal)
        return ModelOutput(self.head(hidden), None, {})
