from agsm.config import merge

from .evaluate import evaluate


def evaluate_lengths(model, config, device, split="test"):
    """Measure task performance using reproducible splits at each requested length."""
    results = []
    for length in config["evaluation"]["sequence_lengths"]:
        variant = merge(config, {"data": {"sequence_length": length}})
        results.append(evaluate(model, variant, device, split))
    return results
