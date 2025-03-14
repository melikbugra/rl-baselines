import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork
from rl_baselines.utils.base_classes.base_sac_neural_network import BaseSACNeuralNetwork


class SACNetworkMLP(BaseSACNeuralNetwork):
    def __init__(
        self,
        actor_mlp: BaseNeuralNetwork,
        critic1_mlp: BaseNeuralNetwork,
        critic2_mlp: BaseNeuralNetwork,
        target_critic1_mlp: BaseNeuralNetwork,
        target_critic2_mlp: BaseNeuralNetwork,
    ):
        super().__init__()
        self.network_type: str = "mlp"
        self.actor = actor_mlp
        self.critic1 = critic1_mlp
        self.critic2 = critic2_mlp
        self.target_critic1 = target_critic1_mlp
        self.target_critic2 = target_critic2_mlp

        self.action_type = actor_mlp.action_type
        self.action_dim = actor_mlp.action_dim

    def forward(
        self,
        state: Tensor,
        action: Tensor = None,
        actor_pass: bool = False,
        critic_pass: bool = False,
        target_pass: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if actor_pass:
            outs = self.actor(state)
        else:
            outs = None

        if critic_pass:
            q1 = self.critic1(torch.cat([state, action], dim=-1))[0]
            q2 = self.critic2(torch.cat([state, action], dim=-1))[0]
        else:
            q1 = None
            q2 = None

        if target_pass:
            target_q1 = self.target_critic1(torch.cat([state, action], dim=-1))[0]
            target_q2 = self.target_critic2(torch.cat([state, action], dim=-1))[0]
        else:
            target_q1 = None
            target_q2 = None

        return outs, q1, q2, target_q1, target_q2
