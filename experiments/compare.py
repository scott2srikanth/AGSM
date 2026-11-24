import argparse
import json
from pathlib import Path

from evaluation.statistical_analysis import paired_bootstrap
from tracking.artifact_manager import write_json
from tracking.report import compare_runs


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate reports and analyze paired seeds"
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", default="A8")
    parser.add_argument("--metric", default="accuracy")
    args = parser.parse_args()
    directories = sorted(
        path.parent for path in args.directory.glob("*/experiment_summary.json")
    )
    if not directories:
        parser.error("No child experiments have a completed summary")
    compare_runs(directories, args.directory / "ablation_comparison.csv")
    groups = {}
    for path in directories:
        label = path.name.rsplit("-seed", 1)[0]
        summary = json.loads((path / "experiment_summary.json").read_text())
        seed = summary["metadata"]["config"]["training"]["seed"]
        score = summary["test"]["metrics"].get(args.metric)
        if score is None:
            continue
        if seed in groups.setdefault(label, {}):
            raise ValueError(f"Duplicate seed {seed} for variant {label}")
        groups[label][seed] = score
    reference = groups.get(args.reference, {})
    comparisons = {}
    for label, scores in groups.items():
        if label == args.reference:
            continue
        comparisons[label] = (
            paired_bootstrap(reference, scores)
            if (len(reference) > 1 and reference.keys() == scores.keys())
            else None
        )
    write_json(
        args.directory / "paired_statistics.json",
        {
            "reference": args.reference,
            "metric": args.metric,
            "difference": "reference minus variant",
            "comparisons": comparisons,
        },
    )


if __name__ == "__main__":
    main()
