import torch

from training.losses import compute_losses


def test_joint_gradient_flow(model):
    tokens = torch.randint(0, 16, (3, 24))
    output = model(tokens)
    losses = compute_losses(
        output,
        tokens.roll(-1, 1),
        {
            "compute_weight": 0.1,
            "write_weight": 0.1,
            "stability_weight": 0.1,
            "fast_threshold": 0.0,
            "slow_threshold": 0.0,
        },
    )
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses["total"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert torch.all(model.controller.operations.weight.grad.abs().sum(dim=1) > 0)
