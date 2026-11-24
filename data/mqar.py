import torch


def generate(tokens, vocab_size, rng):
    """Synthetic MQAR variant with separate key/value alphabets and shuffled queries."""
    midpoint = 3 + (vocab_size - 3) // 2
    count = min(tokens.numel() // 6, midpoint - 3)
    keys = torch.randperm(midpoint - 3, generator=rng)[:count] + 3
    values = torch.randint(midpoint, vocab_size, (count,), generator=rng)
    tokens[: 2 * count : 2] = keys
    tokens[1 : 2 * count : 2] = values
    tokens[2 * count : -2 * count] = torch.randint(
        midpoint, vocab_size, (tokens.numel() - 4 * count,), generator=rng
    )
    order = torch.randperm(count, generator=rng)
    tokens[-2 * count :: 2] = 1
    tokens[-2 * count + 1 :: 2] = keys[order]
    targets = torch.full_like(tokens, -100)
    targets[-2 * count + 1 :: 2] = values[order]
    return tokens, targets
