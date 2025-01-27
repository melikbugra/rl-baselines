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
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        if isinstance(output_neurons, int):
            self.action_type = "discrete"
            self.action_dim = 1
            self.actor_head = nn.Linear(512, output_neurons)
            self.critic_head = nn.Linear(512, 1)

        elif isinstance(output_neurons, list):
            self.action_type = "multidiscrete"
            self.action_dim = len(output_neurons)
            self.actor_heads: list[nn.Linear] = []
            for output_neuron in output_neurons:
                self.actor_heads.append(nn.Linear(512, output_neuron))
            self.critic_head = nn.Linear(512, 1)

        elif isinstance(output_neurons, tuple):
            self.action_type = "continuous"
            self.action_dim = output_neurons[0]
            self.actor_mean_layers: list[nn.Linear] = []
            self.actor_log_std_layers: list[nn.Linear] = []
            for output_neuron in output_neurons:
                self.actor_mean_layers.append(nn.Linear(512, output_neuron))
                self.actor_log_std_layers.append(nn.Linear(512, output_neuron))
            self.critic_head = nn.Linear(512, 1)

        self.to(device)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, state: Tensor):
        x = self.conv(state).view(state.size()[0], -1)
        x = self.fc(x)

        if self.action_type == "discrete":
            action_probs = self.actor_head(x)
            state_value = self.critic_head(x)
            return [action_probs], state_value

        elif self.action_type == "multidiscrete":
            action_probs: list[Tensor] = []
            for head in self.actor_heads:
                action_probs.append(head(x))
            state_value = self.critic_head(x)
            return action_probs, state_value

        elif self.action_type == "continuous":
            actions: list[tuple[Tensor, Tensor]] = []
            for i in range(len(self.actor_mean_layers)):
                mean = self.actor_mean_layers[i](x)
                log_std = self.actor_log_std_layers[i](x)
                std = torch.exp(log_std)
                actions.append((mean, std))
            state_value = self.critic_head(x)
            return actions, state_value

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain("relu")
                )
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
