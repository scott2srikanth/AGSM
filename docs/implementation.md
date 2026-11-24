# Streaming and export

```python
import torch
from agsm.config import build_model, load_config

model = build_model(load_config("configs/model/agsm_tiny.yaml")).eval()
with torch.inference_mode():
    state = model.initialize_state(batch_size=1)
    result = model.process_chunk(torch.tensor([[4, 8, 1]]), state)
    state = model.get_state(result)
    result = model.process_chunk(torch.tensor([[5, 1]]), state)
    state = model.reset_state(model.get_state(result))
```

The caller owns and passes back the two recurrent tensors. Use `RecurrentState.detach()` at explicit truncated-backpropagation boundaries. The built-in trainer backpropagates through each complete training example.

Export a single guarded token update with explicit fast and slow tensors.

```bash
python -m scripts.export_model runs/preflight/best.pt --output runs/preflight/step.ts
python -m scripts.export_model runs/preflight/best.pt --format onnx --output runs/preflight/step.onnx
python -m scripts.export_model runs/preflight/best.pt --int8-linear --output runs/preflight/step-int8.ts
```

Install `.[export]` for ONNX validation. Inputs are token IDs [B], fast state [B,V,K], slow state [B,H], and a boolean validity flag [B]. Outputs are logits [B,C] and next-state tensors. Export uses dense soft refinement for static graph portability. Compare its logits with `model(tokens, compact=False)`. The INT8 option dynamically quantizes linear layers while retaining recurrent state at its configured precision.

ONNX Runtime Mobile and Core ML deployment require runtime operator and numerical validation, packaging, and measurement on a physical device. The desktop CPU results cover reference runtime behavior. FP16 and low-precision memory storage are experimental and require device-specific numerical tests.
