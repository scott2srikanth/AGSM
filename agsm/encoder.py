from torch import Tensor, nn


class InputEncoder(nn.Module):
    """Embed token IDs and project to the recurrent model width."""

    def __init__(self, vocab_size: int, embedding_dim: int, model_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, model_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projection(self.embedding(tokens))
