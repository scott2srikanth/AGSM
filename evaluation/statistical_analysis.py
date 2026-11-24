import math
import random
import statistics


def paired_bootstrap(first, second, seed=42, repeats=10000):
    """Bootstrap the mean paired difference across matched independent seeds."""
    if first.keys() != second.keys() or len(first) < 2 or repeats < 100:
        raise ValueError(
            "Require matched seed keys, two or more seeds, and at least 100 resamples"
        )
    differences = [first[key] - second[key] for key in sorted(first)]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("Scores must be finite")
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(rng.choices(differences, k=len(differences)))
        for _ in range(repeats)
    )
    return {
        "mean_paired_difference": statistics.mean(differences),
        "bootstrap_ci_95": [
            samples[int(0.025 * repeats)],
            samples[int(0.975 * repeats) - 1],
        ],
        "pairs": len(differences),
        "bootstrap_seed": seed,
        "resamples": repeats,
    }
