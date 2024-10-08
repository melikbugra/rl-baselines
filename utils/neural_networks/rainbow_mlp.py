import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from utils.base_classes.base_neural_network import BaseNeuralNetwork
from utils.neural_networks.layers import NoisyLinear


class RainbowMLP(BaseNeuralNetwork):
    def __init__(
        self,
        input_neurons: int,
        network_arch: list[int],
        output_neurons: list[int] | int,
        noisy: bool = False,
        dueling: bool = False,
        device: torch.device = "cpu",
    ):
        super().__init__()

        self.network_type: str = "mlp"

        fc_num = len(network_arch)

        self.layer_neuron_nums: list[int] = [input_neurons] + network_arch

        self.fc_list: list[nn.Linear] | list[NoisyLinear] = nn.ModuleList()

        self.fc_list.append(
            nn.Linear(self.layer_neuron_nums[0], self.layer_neuron_nums[1])
        )

        for i in range(1, fc_num):
            if noisy:
                self.fc_list.append(
                    NoisyLinear(
                        self.layer_neuron_nums[i], self.layer_neuron_nums[i + 1]
                    )
                )
            else:
                self.fc_list.append(
                    nn.Linear(self.layer_neuron_nums[i], self.layer_neuron_nums[i + 1])
                )

        if isinstance(output_neurons, int):
            self.action_type = "discrete"
            self.action_dim = 1
            if noisy:
                self.head = NoisyLinear(self.layer_neuron_nums[-1], output_neurons)
            else:
                self.head = nn.Linear(self.layer_neuron_nums[-1], output_neurons)

        elif isinstance(output_neurons, list):
            self.action_type = "multidiscrete"
            self.action_dim = len(output_neurons)

            if noisy:
                self.heads: list[NoisyLinear] = []
                for output_neuron in output_neurons:
                    self.heads.append(
                        NoisyLinear(self.layer_neuron_nums[-1], output_neuron)
                    )
            else:
                self.heads: list[nn.Linear] = []
                for output_neuron in output_neurons:
                    self.heads.append(
                        nn.Linear(self.layer_neuron_nums[-1], output_neuron)
                    )

        # self._initialize_weights()
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

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain("relu")
                )
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def reset_noise(self):
        """Reset all noisy layers."""
        for layer in self.fc_list[1:]:
            layer.reset_noise()
