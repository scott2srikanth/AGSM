import torch

from agsm.refinement import AdaptiveRefinement


def test_expected_depth_and_real_active_batch_compaction():
    engine = AdaptiveRefinement(4, 4, 8, 3)
    features = torch.randn(2, 4)
    probabilities = torch.tensor([[0.95], [0.2]], requires_grad=True)
    halt = lambda context, z: context  # noqa: E731
    training = engine(features, probabilities, halt, 0.9)
    expected = torch.tensor(
        [0.95 + 2 * 0.05 * 0.95 + 3 * 0.05**2, 0.2 + 2 * 0.8 * 0.2 + 3 * 0.8**2]
    )
    torch.testing.assert_close(training.expected, expected)
    training.expected.sum().backward()
    assert probabilities.grad.abs().min() > 0
    engine.eval()
    sizes = []
    with engine.delta.register_forward_pre_hook(
        lambda module, args: sizes.append(len(args[0]))
    ):
        inference = engine(features, probabilities, halt, 0.9)
    assert sizes == [2, 1, 1]
    assert inference.executed.tolist() == [1, 3]
    torch.testing.assert_close(inference.expected[1], expected[1])
