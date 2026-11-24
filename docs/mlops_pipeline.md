# Research MLOps

## GitHub

Keep stable code on `main`. Use a feature branch and pull request for architecture changes. Configure a branch ruleset to require the CPU validation, lint, and unit-test jobs on GitHub. CI runs a synthetic CPU experiment, validates YAML, exercises gradients, builds the package, and imports the installed wheel outside the checkout.

Push a version tag such as `v0.1.0` to trigger a GitHub release for the source distribution and wheel. Upload selected checkpoint artifacts explicitly after checking their provenance and redistribution terms. Store ordinary experiment checkpoints and datasets in Drive or an artifact store. Use Git LFS only if the repository owner decides to version a large asset and has sufficient quota. Keep configuration, small summaries, and links in Git.

## Colab and restart safety

The notebooks default to the AGSM GitHub repository and a CI-verified commit. Set `AGSM_REPO_URL` and `AGSM_GIT_REF` before running setup to override those defaults. Mount Drive and choose a unique run name. Notebook 03 uses the base training preset and offline W&B. For online tracking, change the resolved tracking mode and load `WANDB_API_KEY` from Colab Secrets before launching the subprocess. Keep credentials out of YAML.

A checkpoint includes resolved settings, model, optimizer, scheduler, mixed-precision scaler, RNG state, metric history, best weights, and early-stopping counters. Checkpoints are atomically replaced at saved epoch boundaries. Restart with the same settings, hardware mode, source revision, and output path. A disconnect can require replaying unsaved epochs. If the latest file is unreadable, resume scans older saved epochs.

## Metadata

`metadata.json` records configuration, Git commit and dirty status, dataset details, parameter count, and runtime. A checkout without a commit records a null revision. `metrics.json` contains per-epoch training and validation results. W&B optionally mirrors epoch metrics and run metadata. The experiment report stores separate post-training test and benchmark artifacts locally or in Drive.
