import numpy as np
import torch.nn as nn
import torch

from utils.base_classes.base_neural_network import BaseNeuralNetwork
from utils.neural_networks.layers import NoisyLinear


class RainbowCNN(BaseNeuralNetwork):
    def __init__(
        self,
        input_shape: list[int],
        output_neurons: int,
        noisy: bool = False,
        device: torch.device = "cpu",
    ):
        super().__init__()

        self.network_type: str = "cnn"

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        if noisy:
            self.linear_layers: list[NoisyLinear] = [
                NoisyLinear(conv_out_size, 512),
                NoisyLinear(512, output_neurons),
            ]
        else:
            self.linear_layers: list[nn.Linear] = [
                nn.Linear(conv_out_size, 512),
                nn.Linear(512, output_neurons),
            ]

        self.fc = nn.Sequential(self.linear_layers[0], nn.ReLU(), self.linear_layers[1])

        self.to(device)

        self.action_type = "discrete"
        self.action_dim = 1

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        action_values = self.fc(conv_out)
        return [
            action_values
        ]  # we return this as a list to be compliant with the multidiscrete case, because it returns also a list

    def reset_noise(self):
        """Reset all noisy layers."""
        for layer in self.linear_layers:
            layer.reset_noise()
