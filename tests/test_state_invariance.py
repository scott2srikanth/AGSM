import pytest
import torch

from agsm import AGSM


@pytest.mark.parametrize("length", [128, 512, 2048, 4096, 8192])
@torch.inference_mode()
def test_state_stays_fixed_and_finite(length):
    torch.manual_seed(7)
    model = AGSM(8, 4, 3, 2, 4, 4, 6, 1).eval()
    output = model(torch.randint(0, 8, (1, length)))
    assert output.state.nbytes == model.initialize_state(1).nbytes == 40
    assert output.state.fast.shape == (1, 2, 3)
    assert output.state.slow.shape == (1, 4)
    assert torch.isfinite(output.logits).all()
    for diagnostic in output.diagnostics.values():
        assert torch.isfinite(diagnostic).all()
    assert output.diagnostics["fast_norm"].max() <= 16.00001
