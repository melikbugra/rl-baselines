import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import numpy as np

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork


class ActorCriticCNN(BaseNeuralNetwork):
    def __init__(
        self,
        input_shape: list[int],
        output_neurons: int | list[int] | tuple[int],
        device: torch.device = "cpu",
    ):
        super().__init__()

        self.network_type: str = "cnn"

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        if isinstance(output_neurons, int):
            self.action_type = "discrete"
            self.action_dim = 1
            self.actor_head = nn.Sequential(
                nn.Linear(conv_out_size, 128), nn.ReLU(), nn.Linear(128, output_neurons)
            )

            self.critic_head = nn.Sequential(
                nn.Linear(conv_out_size, 128), nn.ReLU(), nn.Linear(128, 1)
            )
        elif isinstance(output_neurons, list):
            self.action_type = "multidiscrete"
            self.action_dim = len(output_neurons)
            self.actor_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(conv_out_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, output_neurons[i]),
                    )
                    for i in range(self.action_dim)
                ]
            )
            self.critic_head = nn.Sequential(
                nn.Linear(conv_out_size, 128), nn.ReLU(), nn.Linear(128, 1)
            )

        elif isinstance(output_neurons, tuple):
            self.action_type = "continuous"
            self.action_dim = output_neurons[0]

            self.actor_head = nn.Sequential(
                nn.Linear(conv_out_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_dim),
            )
            self.log_std = nn.Parameter(torch.zeros(output_neurons))
            self.critic_head = nn.Sequential(
                nn.Linear(conv_out_size, 128), nn.ReLU(), nn.Linear(128, 1)
            )

        self.to(device)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, state: Tensor):
        x = state.squeeze(1)
        x = self.conv(x).view(x.size()[0], -1)
        # x = self.fc(x)

        if self.action_type == "discrete":
            action_probs = self.actor_head(x)
            state_value = self.critic_head(x)
            return [action_probs], [state_value]

        elif self.action_type == "multidiscrete":
            action_probs: list[Tensor] = []
            for head in self.actor_heads:
                action_probs.append(head(x))
            state_value = self.critic_head(x)
            return action_probs, state_value

        elif self.action_type == "continuous":
            outs: list[tuple[Tensor, Tensor]] = []
            mean = self.actor_head(x)
            std = torch.exp(self.log_std)
            outs.append((mean, std))

            state_value = self.critic_head(x)

            return outs, [state_value]

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain("relu")
                )
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
