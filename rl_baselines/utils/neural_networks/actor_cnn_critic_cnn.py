import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork


class ActorCNNCriticCNN(BaseNeuralNetwork):
    def __init__(
        self,
        actor_cnn: BaseNeuralNetwork,
        critic_cnn: BaseNeuralNetwork,
    ):
        super().__init__()
        self.actor_cnn = actor_cnn
        self.critic_cnn = critic_cnn

        self.action_type = actor_cnn.action_type
        self.action_dim = critic_cnn.action_dim

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        action_probs = self.actor_cnn(x)
        value = self.critic_cnn(x)

        return action_probs, value
