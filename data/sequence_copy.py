import torch


def generate(tokens, vocab_size, rng):
    """Emit the initial block at the final consecutive query markers."""
    width = max(1, tokens.numel() // 4)
    targets = torch.full_like(tokens, -100)
    tokens[-width:] = 1
    targets[-width:] = tokens[:width]
    return tokens, targets
