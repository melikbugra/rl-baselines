from abc import ABC
from typing import Any

import torch.nn as nn
from torch import Tensor


class BaseNeuralNetwork(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

        self.action_type: str
        self.action_dim: int
        self.network_type: str

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return super().__call__(*args, **kwds)
