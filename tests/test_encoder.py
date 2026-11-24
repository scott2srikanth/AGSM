import torch

from agsm.encoder import InputEncoder
from agsm.selective_gate import SelectiveInformationGate


def test_encoder_and_qkv_dimensions():
    encoder = InputEncoder(16, 6, 8)
    x = encoder(torch.zeros(2, 4, dtype=torch.long))
    assert x.shape == (2, 4, 8)
    result = SelectiveInformationGate(8, 7, 4, 5)(x[:, 0], torch.zeros(2, 7))
    assert [tensor.shape for tensor in result] == [(2, 8), (2, 4), (2, 4), (2, 5)]
