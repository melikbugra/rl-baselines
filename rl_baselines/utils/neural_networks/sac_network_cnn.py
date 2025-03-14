import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork
from rl_baselines.utils.base_classes.base_sac_neural_network import BaseSACNeuralNetwork
from rl_baselines.utils.neural_networks.cnn import CNN


class SACNetworkCNN(BaseSACNeuralNetwork):
    def __init__(
        self,
        actor_cnn: CNN,
        critic1_cnn: CNN,
        critic2_cnn: CNN,
        target_critic1_cnn: CNN,
        target_critic2_cnn: CNN,
    ):
        super().__init__()
        self.network_type: str = "cnn"
        self.actor: CNN = actor_cnn
        self.critic1: CNN = critic1_cnn
        self.critic2: CNN = critic2_cnn
        self.target_critic1: CNN = target_critic1_cnn
        self.target_critic2: CNN = target_critic2_cnn

        self.action_type = actor_cnn.action_type
        self.action_dim = actor_cnn.action_dim

        self.conv_out_size = self.critic1._get_conv_out_size(actor_cnn.input_shape)
        self.critic1.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + self.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        ).to(actor_cnn.device)
        self.critic2.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + self.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        ).to(actor_cnn.device)
        self.target_critic1.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + self.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        ).to(actor_cnn.device)
        self.target_critic2.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + self.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        ).to(actor_cnn.device)

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
            state = state.squeeze(1)

            conv_out1 = self.critic1.conv(state).view(state.size()[0], -1)
            conv_out2 = self.critic2.conv(state).view(state.size()[0], -1)

            q1 = self.critic1.fc(torch.cat([conv_out1, action], dim=1))
            q2 = self.critic2.fc(torch.cat([conv_out2, action], dim=1))

        else:
            q1 = None
            q2 = None

        if target_pass:
            state = state.squeeze(1)

            conv_out1 = self.target_critic1.conv(state).view(state.size()[0], -1)
            conv_out2 = self.target_critic2.conv(state).view(state.size()[0], -1)

            target_q1 = self.target_critic1.fc(torch.cat([conv_out1, action], dim=1))
            target_q2 = self.target_critic2.fc(torch.cat([conv_out2, action], dim=1))
        else:
            target_q1 = None
            target_q2 = None

        return outs, q1, q2, target_q1, target_q2
