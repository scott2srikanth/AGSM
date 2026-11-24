import pytest
import torch

from agsm import AGSM


@pytest.fixture(autouse=True, scope="session")
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return AGSM(16, 8, 4, 5, 7, 8, 12, 3)
