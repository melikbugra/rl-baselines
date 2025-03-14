import numpy as np
import torch.nn as nn
import torch
from torch import Tensor

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork


class CNN(BaseNeuralNetwork):
    def __init__(
        self,
        input_shape: list[int],
        output_neurons: int,
        device: torch.device,
    ):
        super().__init__()

        self.network_type: str = "cnn"
        self.device = device

        self.input_shape = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.conv.to(device)

        conv_out_size = self._get_conv_out_size(input_shape)

        if isinstance(output_neurons, int):
            self.action_type = "discrete"
            self.action_dim = 1
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, output_neurons)
            )
        elif isinstance(output_neurons, list):
            self.action_type = "multidiscrete"
            self.action_dim = len(output_neurons)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(conv_out_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, output_neurons[i]),
                    )
                    for i in range(self.action_dim)
                ]
            )
        elif isinstance(output_neurons, tuple):
            self.action_type = "continuous"
            self.action_dim = np.prod(output_neurons)
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, self.action_dim),
            )
            self.std_head = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, self.action_dim),
            )

        self.to(device)

    def _get_conv_out_size(self, shape):
        o = self.conv(torch.zeros(1, *shape, device=self.device))
        return int(np.prod(o.size()))

    def forward(self, x: Tensor):
        x = x.squeeze(1)
        conv_out = self.conv(x).view(x.size()[0], -1)

        if self.action_type == "discrete":
            action_values = self.fc(conv_out)

            return [action_values]

        elif self.action_type == "multidiscrete":
            sub_action_values: list[Tensor] = []

            for head in self.heads:
                sub_action_values.append(head(conv_out))

            return sub_action_values

        elif self.action_type == "continuous":
            outs: list[tuple[Tensor, Tensor]] = []

            mean = self.fc(conv_out)
            log_std = self.std_head(conv_out)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            outs.append((mean, std))

            return outs
