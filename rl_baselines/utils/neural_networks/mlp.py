import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import numpy as np

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork


class MLP(BaseNeuralNetwork):
    def __init__(
        self,
        input_neurons: int,
        network_arch: list[int],
        output_neurons: list[int] | int,
        device: torch.device,
    ):
        super().__init__()

        self.network_type: str = "mlp"

        fc_num = len(network_arch)

        self.layer_neuron_nums: list[int] = [input_neurons] + network_arch

        self.fc_list = nn.ModuleList()

        for i in range(fc_num):
            self.fc_list.append(
                nn.Linear(self.layer_neuron_nums[i], self.layer_neuron_nums[i + 1])
            )

        if isinstance(output_neurons, int):
            self.action_type = "discrete"
            self.action_dim = 1
            self.head = nn.Linear(self.layer_neuron_nums[-1], output_neurons)

        elif isinstance(output_neurons, list):
            self.action_type = "multidiscrete"
            self.action_dim = len(output_neurons)

            self.heads: list[nn.Linear] = []
            for output_neuron in output_neurons:
                self.heads.append(nn.Linear(self.layer_neuron_nums[-1], output_neuron))

        elif isinstance(output_neurons, tuple):
            self.action_type = "continuous"
            self.action_dim = np.prod(output_neurons)

            self.mean_head = nn.Linear(self.layer_neuron_nums[-1], self.action_dim)
            self.std_head = nn.Linear(self.layer_neuron_nums[-1], self.action_dim)

        self.apply(self._initialize_weights)
        self.to(device)

    def forward(self, state: Tensor):
        x = state
        for i in range(len(self.fc_list)):
            x = F.relu(self.fc_list[i](x))

        if self.action_type == "discrete":
            action_values = self.head(x)
            return [
                action_values
            ]  # we return this as a list to be compliant with the multidiscrete case, because it returns also a list

        elif self.action_type == "multidiscrete":
            sub_action_values: list[Tensor] = []

            for head in self.heads:
                sub_action_values.append(head(x))

            return sub_action_values

        elif self.action_type == "continuous":
            outs: list[tuple[Tensor, Tensor]] = []

            mean = self.mean_head(x)
            log_std = self.std_head(x)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            outs.append((mean, std))

            return outs

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
