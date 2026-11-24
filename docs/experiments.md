# Experiment protocol

Choose the training task, model variants, random seeds, length grid, and compute budget before final test evaluation. Use validation task loss for model selection and rerun each selected configuration across independent, paired seeds. Record deviations from planned parameter and compute budgets.

## Datasets

The deterministic synthetic generators implement delayed recall, copying, variable-lag recall, and a synthetic MQAR variant. The MQAR setup samples unique keys and shuffles queries. Its format and scores are specific to this implementation.

For ListOps, stage the [LRA release](https://github.com/google-research/long-range-arena) TSV splits and record the upstream revision in `configs/training/listops.yaml`. The loader filters examples beyond the configured length and left pads retained examples. Report this filter with results.

For WikiText-103, stage raw text splits from the [documented dataset source](https://huggingface.co/datasets/Salesforce/wikitext) and pin its revision in `configs/training/wikitext103.yaml`. When using the parquet release, write the `text` column in original split order with one newline between rows and record the export procedure. The model uses UTF-8 bytes and reports byte cross-entropy. Standard word-level WikiText perplexity uses different units.

Use [NVIDIA RULER](https://github.com/NVIDIA/RULER) to generate pinned JSONL inputs with the chosen task, tokenizer, template and context length. With a trained UTF-8 language model, generate predictions using `python -m evaluation.ruler CHECKPOINT INPUT_JSONL OUTPUT_JSONL --max-new-bytes N`. Place the output under RULER's expected task and subset paths and use the pinned upstream evaluator. Record byte counts alongside upstream token counts.

## Baselines and ablations

GRU and causal Transformer baselines run on CPU. Mamba-2 uses [state-spaces/mamba](https://github.com/state-spaces/mamba). RWKV-7 and Gated DeltaNet use [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention). Follow each upstream install guide for your CUDA, PyTorch and Python versions, then pin the working revision in your environment lock. The adapters have not been validated on a GPU in this workspace. On an incompatible runtime, record the missing run and use a separate compatible CUDA runtime for that baseline. AGSM 1.0 requires its original code or a versioned specification.

A0 runs the GRU. Memory and controller YAML files define A2–A11. A4–A7 keep retention, slow update, and update strength fixed at one and progressively enable write gating, retrieval scaling and adaptive depth. A8 enables all joint controller gates. A9 disables the residual feature. A10 forces full depth. A11 sets the write penalty to zero. A1 is pending the original AGSM 1.0.

```bash
python -m experiments.capacity_sweep --output runs/parameter-plan.json
python -m experiments.capacity_sweep --flop-budgets 100000000 300000000 --length 128 --output runs/compute-plan.json
python -m experiments.ablation_runner --config configs/experiments/controller_ablation.yaml --output runs/controller-v1 --seeds 11 22 33
```

Capacity matching selects candidate widths and records the signed deviation. Compute matching uses the operations covered by PyTorch Profiler. Report coverage and validate operator comparability before interpreting budget matches. Long-context metrics are evaluated at the YAML length grid. Peak CPU RSS is a process lifetime high-water mark. GPU peak reports allocator memory, including allocations beyond recurrent state. An executed-depth distribution should be read alongside synchronized latency measurements.

Run `python -m experiments.compare runs/controller-v1 --reference A8` to generate a consolidated table and paired bootstrap differences. The command expects the variant rows to use independently chosen, matching seeds and comparable datasets and budgets. Record the sample size and uncertainty, and keep negative results. Smoke runs establish an executable pipeline. Claims about accuracy or efficiency require full controlled comparisons.

## Local pilot commands

```bash
python -m experiments.ablation_runner --config configs/experiments/cpu_pilot.yaml --output runs/paired-replay --variants A0 transformer A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 --seeds 11 22 33
python -m experiments.compare runs/paired-replay --reference A8
python -m experiments.capacity_sweep --config configs/experiments/budget_pilot.yaml --output runs/parameter-replay.json
python -m experiments.run_budget runs/parameter-replay.json --output runs/parameter-replay
python -m experiments.capacity_sweep --config configs/experiments/budget_pilot.yaml --flop-budgets 2000000 --length 8 --output runs/compute-replay.json
python -m experiments.run_budget runs/compute-replay.json --output runs/compute-replay
```

The paired pilot uses 128 training examples per seed and three epochs. Its output grid is 16, 32, and 128 tokens. The budget pilot uses a single epoch and four held-out examples per condition to exercise the matched-run workflow.
