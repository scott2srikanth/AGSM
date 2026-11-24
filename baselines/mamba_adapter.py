import torch
from torch import nn

from agsm.model import ModelOutput


class Mamba2Adapter(nn.Module):
    """Task adapter for the upstream state-spaces Mamba-2 block."""

    def __init__(self, vocab_size, model_dim, output_dim=None, **_):
        super().__init__()
        if model_dim % 32:
            raise ValueError("Mamba-2 model_dim must be divisible by 32")
        from mamba_ssm import Mamba2

        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.core = Mamba2(d_model=model_dim, headdim=64)
        self.head = nn.Linear(model_dim, output_dim or vocab_size)

    def initialize_state(self, batch_size):
        return self.core.allocate_inference_cache(batch_size, max_seqlen=1)

    def forward(self, tokens, state=None, mask=None):
        if tokens.device.type != "cuda":
            raise RuntimeError("The Mamba-2 adapter requires a supported CUDA runtime")
        if mask is not None:
            raise ValueError("Use unpadded, same-length batches for Mamba-2")
        x = self.embedding(tokens)
        if state is None:
            return ModelOutput(self.head(self.core(x)), None, {})
        if self.training:
            raise ValueError("Mamba-2 streaming cache is inference-only")
        conv, recurrent = (item.clone() for item in state)
        steps = []
        for feature in x.split(1, dim=1):
            output, conv, recurrent = self.core.step(feature, conv, recurrent)
            steps.append(output)
        return ModelOutput(self.head(torch.cat(steps, dim=1)), (conv, recurrent), {})
