import torch


def generate(tokens, vocab_size, rng):
    """Recall the first observed symbol at the final query marker."""
    targets = torch.full_like(tokens, -100)
    tokens[-1] = 1
    targets[-1] = tokens[0]
    return tokens, targets
