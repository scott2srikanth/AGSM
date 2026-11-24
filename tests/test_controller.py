import torch


def test_shared_controller_produces_bounded_input_dependent_gates(model):
    features = torch.stack((torch.zeros(27), torch.ones(27)))
    gates, context = model.controller(features)
    assert len(gates) == 5
    for gate in gates.values():
        assert torch.all((gate > 0) & (gate < 1))
        assert gate[0] != gate[1]
    halt = model.controller.halting_probability(context, torch.ones(2, 8))
    assert torch.all((halt > 0) & (halt < 1))
