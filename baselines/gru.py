import torch
from torch import nn

from agsm.model import ModelOutput


class GRUBaseline(nn.Module):
    """Causal GRU with explicit reusable hidden state."""

    def __init__(self, vocab_size, model_dim, output_dim=None, **_):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.gru = nn.GRU(model_dim, model_dim, batch_first=True)
        self.head = nn.Linear(model_dim, output_dim or vocab_size)

    def forward(self, tokens, state=None, mask=None):
        features = self.embedding(tokens)
        if mask is None:
            hidden, state = self.gru(features, state)
        else:
            if state is None:
                state = features.new_zeros(1, features.shape[0], features.shape[2])
            hidden_steps = []
            for step, valid in zip(features.unbind(1), mask.unbind(1)):
                _, candidate = self.gru(step[:, None], state)
                state = torch.where(valid[None, :, None], candidate, state)
                hidden_steps.append(state[0])
            hidden = torch.stack(hidden_steps, dim=1)
        return ModelOutput(self.head(hidden), state, {})
