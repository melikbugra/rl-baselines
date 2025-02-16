import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork


class ActorMLPCriticMLP(BaseNeuralNetwork):
    def __init__(
        self,
        actor_mlp: BaseNeuralNetwork,
        critic_mlp: BaseNeuralNetwork,
    ):
        super().__init__()
        self.actor_mlp = actor_mlp
        self.critic_mlp = critic_mlp

        self.action_type = actor_mlp.action_type
        self.action_dim = actor_mlp.action_dim

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        action_probs = self.actor_mlp(x)
        value = self.critic_mlp(x)

        return action_probs, value
