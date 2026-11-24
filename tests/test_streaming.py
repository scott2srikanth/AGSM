import torch


@torch.inference_mode()
def test_chunking_padding_and_reset(model):
    model.eval()
    tokens = torch.tensor([[3, 5, 6, 1, 2, 9], [1, 11, 4, 7, 8, 5]])
    full = model(tokens)
    first = model.process_chunk(tokens[:, :2])
    second = model.process_chunk(tokens[:, 2:], model.get_state(first))
    torch.testing.assert_close(
        torch.cat((first.logits, second.logits), dim=1), full.logits
    )
    torch.testing.assert_close(second.state.fast, full.state.fast)
    torch.testing.assert_close(second.state.slow, full.state.slow)
    padded = model(
        tokens[:, :2], full.state, torch.zeros_like(tokens[:, :2], dtype=torch.bool)
    )
    torch.testing.assert_close(padded.state.fast, full.state.fast)
    torch.testing.assert_close(padded.state.slow, full.state.slow)
    reset = model(tokens, model.reset_state(full.state))
    torch.testing.assert_close(reset.logits, full.logits)
