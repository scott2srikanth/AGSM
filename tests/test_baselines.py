import pytest
import torch

from agsm.config import build_model, load_config, merge


@pytest.mark.parametrize("name", ["gru", "transformer"])
def test_baseline_forward_backward_and_causality(name):
    config = merge(
        load_config("configs/training/smoke.yaml"), {"model": {"name": name}}
    )
    model = build_model(config)
    tokens = torch.tensor([[0, 0, 1, 5, 3], [4, 5, 6, 7, 8]])
    mask = tokens.ne(0)
    logits = model(tokens, mask=mask).logits
    logits.square().mean().backward()
    assert logits.shape == (2, 5, 64)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    changed = tokens.clone()
    changed[:, -1] = 15
    with torch.inference_mode():
        torch.testing.assert_close(
            model(tokens, mask=mask).logits[:, :-1],
            model(changed, mask=mask).logits[:, :-1],
        )
