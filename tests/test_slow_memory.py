import torch

from agsm.slow_memory import SlowContextualMemory


def test_slow_interpolation_and_reset():
    memory = SlowContextualMemory(3, 2, 4)
    state = memory.initial_state(2)
    encoded, read = torch.ones(2, 3), torch.ones(2, 2)
    candidate = memory.cell(torch.cat((encoded, read), dim=-1), state)
    updated = memory(encoded, read, state, torch.tensor([[0.0], [0.25]]))
    torch.testing.assert_close(updated[0], state[0])
    torch.testing.assert_close(updated[1], candidate[1] * 0.25)
    torch.testing.assert_close(memory.reset_state(updated), state)
