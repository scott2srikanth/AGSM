import torch


def generate(tokens, vocab_size, rng):
    """Recall a symbol at a randomized position explicitly marked in the input."""
    position = int(torch.randint(tokens.numel() - 2, (), generator=rng))
    targets = torch.full_like(tokens, -100)
    tokens[position] = 2
    tokens[-1] = 1
    targets[-1] = tokens[position + 1]
    return tokens, targets
