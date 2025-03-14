from abc import ABC
from typing import Any

import torch.nn as nn
from torch import Tensor

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork


class BaseSACNeuralNetwork(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

        self.action_type: str
        self.action_dim: int
        self.network_type: str
        self.training: bool = True

        self.actor: BaseNeuralNetwork = None
        self.critic1: BaseNeuralNetwork = None
        self.critic2: BaseNeuralNetwork = None
        self.target_critic1: BaseNeuralNetwork = None
        self.target_critic2: BaseNeuralNetwork = None
        self.critic: BaseNeuralNetwork = None

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return super().__call__(*args, **kwds)

    def _get_conv_out_size(self, input_shape: list[int]) -> int:
        raise NotImplementedError
