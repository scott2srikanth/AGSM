# AGSM 2.0

Adaptive Gated State Memory is an experimental, attention-free sequence model. It combines a fixed-size associative matrix and GRU context with a shared controller for memory operations and refinement depth. This repository supports controlled sequence experiments in PyTorch.

## Local preflight

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
python -m pytest -q
python -m scripts.validate_config
python -m scripts.smoke_test
```

## Train, resume, and evaluate

```bash
python -m experiments.run_experiment --config configs/training/smoke.yaml --output runs/preflight
python -m experiments.run_experiment --config configs/training/smoke.yaml --output runs/preflight --resume
python -m scripts.evaluate runs/preflight/best.pt --output runs/preflight/test.json
python -m scripts.benchmark --checkpoint runs/preflight/best.pt --lengths 8 16 --profile --output runs/preflight/profile.json
```

Use a fresh output directory for each configuration and seed. The experiment command trains, evaluates the best validation checkpoint, measures latency, and produces JSON, CSV, Markdown, and PNG artifacts. Use `scripts.train` for training alone. For research, start with [the experiment protocol](docs/experiments.md) before opening the held-out test set.

## Colab

Open [environment setup](notebooks/01_environment_setup.ipynb) and [AGSM training](notebooks/03_agsm_training.ipynb) from a GitHub checkout in Colab. Select a GPU runtime. The setup cell checks out a CI-verified AGSM commit by default. Training saves checkpoint epochs to Google Drive and resumes from the latest readable epoch when you reuse the run name and resolved configuration. A runtime without CUDA uses CPU smoke settings.

## Documentation

- [Architecture](docs/architecture.md)
- [Mathematical specification](docs/mathematical_specification.md)
- [Streaming and export](docs/implementation.md)
- [GitHub, Colab, and tracking](docs/mlops_pipeline.md)
- [Datasets, baseline comparisons, and ablations](docs/experiments.md)
- [Research log](docs/research_log.md)
