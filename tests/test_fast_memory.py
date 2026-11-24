import pytest
import torch

from agsm.fast_memory import FastAssociativeMemory


@pytest.mark.parametrize("precision", ["float32", "float64", "float16", "bfloat16"])
def test_delta_update_reset_and_read_before_write(precision):
    memory = FastAssociativeMemory(3, 2, precision)
    state = memory.initial_state(1)
    key = memory.address(torch.tensor([[2.0, 0.0]]))
    value = torch.tensor([[1.0, -2.0, 3.0]]).to(key.dtype)
    strength = torch.ones(1, 1)
    torch.testing.assert_close(memory.read(state, key), torch.zeros_like(value))
    first = memory.write(state, key, value, strength, strength)
    assert first.dtype == state.dtype
    torch.testing.assert_close(memory.read(first, key), value)
    second = memory.write(
        first, key, value - memory.read(first, key), strength / 2, strength
    )
    torch.testing.assert_close(memory.read(second, key), value / 2)
    assert memory.reset_state(first).count_nonzero() == 0
    assert memory.address(torch.zeros_like(key)).count_nonzero() == 0


def test_norm_clipping():
    memory = FastAssociativeMemory(3, 2, norm_limit=0.25)
    gate = torch.ones(2, 1)
    updated = memory.write(
        memory.initial_state(2),
        memory.address(torch.ones(2, 2)),
        torch.ones(2, 3),
        gate,
        gate,
    )
    torch.testing.assert_close(memory.norm(updated), torch.full((2,), 0.25))
