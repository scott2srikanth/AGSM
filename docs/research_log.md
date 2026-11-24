# AGSM 2.0 research log

## Initial implementation and CPU pilots

The workspace started on an unborn `main` branch without a GitHub remote or an AGSM 1.0 implementation. Local runs therefore record a null source commit and a dirty workspace. The selected initial repository license is MIT.

| Phase | Verified locally | Remaining work |
| --- | --- | --- |
| Foundation | Package build and wheel import outside the checkout. Four GitHub Actions workflows passed actionlint. | Hosted workflow execution and branch rules require a GitHub remote. |
| Architecture | Dual recurrent state, shared sigmoid gates, conditional halting, active-row compaction and independent heads. The base preset has 1,784,870 parameters. | CUDA kernel and reduced-precision profiling. |
| Unit testing | CPU gradients, state invariance through 8,192 tokens, streaming/reset, padding, exact depth weighting and batch compaction. | GPU numerical tests. |
| Training | Synthetic CPU training, validation, durable checkpoint fallback, deterministic resume, offline W&B and report generation. All eight notebook sources compile locally. | Hosted Colab, Drive and CUDA execution. |
| Baselines | GRU and Transformer ran locally. Upstream Mamba-2, RWKV-7 and Gated DeltaNet adapters are included. | Run those adapters on compatible CUDA hardware and obtain the original AGSM 1.0 implementation. |
| Experiments | 36 paired CPU pilot runs across 12 variants. Twelve budget smoke runs exercised parameter and profiler-counted FLOP matching. | Research-scale training and evaluation, external datasets, and verified CUDA baselines. |
| Reporting | Per-run plots and tables, consolidated CPU pilot plots, and paired bootstrap output. | Additional independent seeds and hypothesis-specific analysis. |
| Deployment | FP32 TorchScript and ONNX matched dense eager recurrence for two test batch sizes across multiple tokens. Dynamic INT8 TorchScript produced finite outputs on QNNPACK. | INT8 quality measurements, Core ML and ONNX Runtime Mobile device verification, and energy measurement. |

Use the README preflight commands for phases 1–4, the [experiment protocol](experiments.md) for phases 5–7, and [streaming and export](implementation.md) for phase 8.

### Observed CPU results

Seeds 11, 22 and 33 used a 16-token delayed-recall task, 128 training examples, 64 validation examples, 64 primary test examples and three epochs. Mean held-out accuracy was 0.52% for full AGSM, 2.60% for the GRU, and 1.04% for the causal Transformer. Chance under a uniform guess over the 61 eligible symbols is approximately 1.64%. These short runs provide no evidence that AGSM learned delayed recall. Three seeds give limited uncertainty estimates.

The one-epoch budget pilot exercised the 1M, 3M and 5M targets with three local architectures. The largest absolute parameter deviation was 27,968. At a target of 2 million profiler-counted FLOPs, the 8-token candidates measured 2,023,928 for AGSM, 2,026,080 for GRU, and 1,936,082 for Transformer. Those counts cover the profiler's supported operators. Each budget smoke test used four held-out examples.

### Design record

The initial implementation uses a conditional halting distribution, carries remaining mass to the stopping depth, and computes the slow GRU candidate from the unscaled fast read. The controller uses a log-transformed residual ratio to compress the dynamic range of mixed-precision inputs, while diagnostics report the raw ratio. The dense export ABI intentionally evaluates all refinement steps. Raw CPU pilot results and exact resolved configurations are stored under `runs/paired-cpu-pilot` and `runs/budget-pilot` in this workspace.
