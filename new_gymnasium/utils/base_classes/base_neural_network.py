from typing import Any
import torch.nn as nn
from torch import Tensor


class BaseNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        self.action_type: str
        self.action_num: int

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return super().__call__(*args, **kwds)
