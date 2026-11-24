from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class RecurrentState:
    fast: Tensor
    slow: Tensor

    def detach(self) -> "RecurrentState":
        return RecurrentState(self.fast.detach(), self.slow.detach())

    @property
    def nbytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in (self.fast, self.slow))
