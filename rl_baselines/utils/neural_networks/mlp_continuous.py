import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork


class MLPContinuous(BaseNeuralNetwork):
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
            self.mean_layer = nn.Linear(self.layer_neuron_nums[-1], output_neurons)
            self.log_std_layer = nn.Linear(self.layer_neuron_nums[-1], output_neurons)

        elif isinstance(output_neurons, list):
            self.action_type = "multidiscrete"
            self.action_dim = len(output_neurons)

            self.mean_layers: list[nn.Linear] = []
            self.log_std_layers: list[nn.Linear] = []
            for output_neuron in output_neurons:
                self.mean_layers.append(
                    nn.Linear(self.layer_neuron_nums[-1], output_neuron)
                )
                self.log_std_layers.append(
                    nn.Linear(self.layer_neuron_nums[-1], output_neuron)
                )

        # self._initialize_weights()
        self.to(device)

    def forward(self, state: Tensor):
        x = state
        for i in range(len(self.fc_list)):
            x = F.relu(self.fc_list[i](x))

        if self.action_type == "discrete":
            mean = self.mean_layer(x)
            log_std = self.log_std_layer(x)
            return [
                (mean, log_std)
            ]  # we return this as a list to be compliant with the multidiscrete case, because it returns also a list

        elif self.action_type == "multidiscrete":
            outs: list[tuple[Tensor, Tensor]] = []

            for i in range(len(self.mean_layers)):
                mean = self.mean_layers[i](x)
                log_std = self.log_std_layers[i](x)
                outs.append((mean, log_std))

            return outs

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain("relu")
                )
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
