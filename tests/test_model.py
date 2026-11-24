import pytest
import torch

from agsm import AGSM


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize(
    "task", ["classification", "retrieval", "regression", "language_model"]
)
def test_task_head_shapes(batch, task):
    output_dim = 1 if task == "regression" else 13
    model = AGSM(16, 8, 4, 5, 7, 8, 12, 2, task=task, output_dim=output_dim)
    output = model(torch.zeros(batch, 6, dtype=torch.long))
    assert output.logits.shape == (batch, 6, output_dim)
    assert output.state.fast.shape == (batch, 5, 4)
    assert output.state.slow.shape == (batch, 7)
    assert output.diagnostics["executed_steps"].max() == 2


def test_read_before_write_and_causality(model):
    tokens = torch.tensor([[1, 2, 3, 4]])
    modified = torch.tensor([[1, 2, 5, 6]])
    a, b = model(tokens), model(modified)
    torch.testing.assert_close(a.logits[:, :2], b.logits[:, :2])
    assert a.state.fast.count_nonzero() > 0
