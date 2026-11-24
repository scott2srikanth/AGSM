import csv
import json
from pathlib import Path
from statistics import mean

from .artifact_manager import write_json


def write_table(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key) if row.get(key) is not None else "NA" for key in fields}
            for row in rows
        )


def _plot(path, lines, xlabel, ylabel):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    colors = plt.get_cmap("tab20").colors
    for index, (name, points) in enumerate(lines.items()):
        if points:
            x, y = zip(*points)
            axis.plot(x, y, marker="o", label=name, color=colors[index % len(colors)])
    axis.set(xlabel=xlabel, ylabel=ylabel)
    if axis.has_data():
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small")
    else:
        axis.text(0.5, 0.5, "NA", transform=axis.transAxes, ha="center", va="center")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    directory, metadata, history, test, lengths, benchmark, refinement_counts
):
    directory = Path(directory)
    summary = {
        "metadata": metadata,
        "test": test,
        "long_context": lengths,
        "benchmark": benchmark,
        "refinement_counts": refinement_counts,
        "training_seconds": sum(row["epoch_seconds"] for row in history),
    }
    write_json(directory / "experiment_summary.json", summary)
    write_table(directory / "metrics.csv", history)
    _plot(
        directory / "training_curves.png",
        {
            key: [(row["epoch"], row[key]) for row in history if key in row]
            for key in ("train/loss_task", "validation/loss_task")
        },
        "Epoch",
        "Task loss",
    )
    _plot(
        directory / "accuracy_vs_sequence_length.png",
        {
            "Test": [
                (row["dataset"]["sequence_length"], row["metrics"]["accuracy"])
                for row in lengths
                if "accuracy" in row["metrics"]
            ]
        },
        "Sequence length",
        "Accuracy",
    )
    _plot(
        directory / "latency_comparison.png",
        {
            "Median": [(row["sequence_length"], row["median_ms"]) for row in benchmark],
            "p95": [(row["sequence_length"], row["p95_ms"]) for row in benchmark],
        },
        "Sequence length",
        "Latency (ms)",
    )
    _plot(
        directory / "memory_comparison.png",
        {
            key: [
                (row["sequence_length"], row[key])
                for row in benchmark
                if row.get(key) is not None
            ]
            for key in (
                "persistent_state_bytes",
                "peak_gpu_allocated_bytes",
                "peak_process_rss_bytes",
            )
        },
        "Sequence length",
        "Bytes",
    )
    _plot(
        directory / "refinement_distribution.png",
        {
            "Inference": sorted(
                (int(depth), count) for depth, count in refinement_counts.items()
            )
        },
        "Executed steps",
        "Token count",
    )
    write_table(directory / "ablation_comparison.csv", [comparison_row(summary)])
    score = test["metrics"].get("accuracy", "NA")
    (directory / "experiment_report.md").write_text(
        f"# Experiment {metadata['experiment_id']}\n\n"
        "## Research question\n\nHow does the configured model perform on the held-out sequence task?\n\n"
        f"## Configuration and dataset\n\nSee metadata.json for the resolved YAML settings, "
        f"split metadata and training seed. Dataset is {test['dataset']['name']}.\n\n"
        f"## Results and resource usage\n\nTest accuracy is {score}. "
        f"Training recorded {len(history)} epochs and {summary['training_seconds']:.2f} seconds. "
        "Timing and memory measurements are in experiment_summary.json.\n\n"
        "## Baseline comparisons and limitations\n\n"
        "A single run cannot establish a comparative result. Process RSS is a lifetime high-water mark. "
        "Unavailable fields are null in JSON and NA in figures.\n\n"
        f"## Reproducibility\n\nGit commit {metadata['git']['commit'] or 'NA'}. "
        f"Dirty checkout {metadata['git']['dirty']}. See metadata.json and best.pt.\n",
        encoding="utf-8",
    )


def comparison_row(summary):
    metadata = summary["metadata"]
    return {
        "experiment_id": metadata["experiment_id"],
        "model": metadata["config"]["model"]["name"],
        "seed": metadata["config"]["training"]["seed"],
        "commit": metadata["git"]["commit"],
        "parameters": metadata["parameters"],
        "accuracy": summary["test"]["metrics"].get("accuracy"),
        "task_loss": summary["test"]["metrics"]["loss_task"],
    }


def compare_runs(directories, output):
    summaries = [
        json.loads((Path(path) / "experiment_summary.json").read_text())
        for path in directories
    ]
    rows = [
        {"variant": Path(path).name, **comparison_row(summary)}
        for path, summary in zip(directories, summaries)
    ]
    write_table(output, rows)
    lines = [
        "# Experiment comparison",
        "",
        "Each row represents one run. NA indicates an unavailable measurement.",
        "",
        "| Run | Parameters | Held-out accuracy | Held-out task loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        accuracy = row["accuracy"]
        accuracy = f"{accuracy:.4f}" if accuracy is not None else "NA"
        lines.append(
            f"| {row['variant']} | {row['parameters']:,} | {accuracy} | {row['task_loss']:.4f} |"
        )
    Path(output).with_name("comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    parent = Path(output).parent

    def group_points(extract):
        groups = {}
        for path, summary in zip(directories, summaries):
            variant = Path(path).name.rsplit("-seed", 1)[0]
            points = groups.setdefault(variant, {})
            for x, y in extract(summary):
                points.setdefault(x, []).append(y)
        return {
            variant: [(x, mean(values)) for x, values in sorted(points.items())]
            for variant, points in groups.items()
        }

    _plot(
        parent / "accuracy_vs_sequence_length.png",
        group_points(
            lambda summary: (
                (row["dataset"]["sequence_length"], row["metrics"]["accuracy"])
                for row in summary["long_context"]
                if "accuracy" in row["metrics"]
            )
        ),
        "Sequence length",
        "Mean accuracy",
    )
    for filename, metric, ylabel in (
        ("latency_comparison.png", "median_ms", "Median latency (ms)"),
        ("memory_comparison.png", "persistent_state_bytes", "Persistent state (bytes)"),
    ):
        points = group_points(
            lambda summary: (
                (row["sequence_length"], row[metric])
                for row in summary["benchmark"]
                if row.get(metric) is not None
            )
        )
        _plot(parent / filename, points, "Sequence length", ylabel)
