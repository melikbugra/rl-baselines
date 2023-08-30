import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_neurons: int, fc_neuron_nums: list[int], output_neurons: int):
        super().__init__()
        
        self.layer_neuron_nums = [input_neurons] + fc_neuron_nums
        
        self.fc_list: nn.ModuleList = nn.ModuleList()
        self.bn_list: nn.ModuleList = nn.ModuleList()

        for i in range(len(fc_neuron_nums)):
            self.fc_list.append(nn.Linear(self.layer_neuron_nums[i], self.layer_neuron_nums[i+1]))
            fan_in = 1./np.sqrt(self.fc_list[i].weight.data.size()[0])
            self.fc_list[i].weight.data.uniform_(-fan_in, fan_in)
            self.fc_list[i].bias.data.uniform_(-fan_in, fan_in)

        for i in range(len(fc_neuron_nums)):
            self.bn_list.append(nn.LayerNorm(self.layer_neuron_nums[i+1]))

        self.mu = nn.Linear(self.layer_neuron_nums[-1], output_neurons)

        fan_in_mu = 0.003
        self.mu.weight.data.uniform_(-fan_in_mu, fan_in_mu)
        self.mu.bias.data.uniform_(-fan_in_mu, fan_in_mu)

    def forward(self, state):
        x = state
        for i in range(len(self.fc_list)):
            x = self.fc_list[i](x)
            x = F.relu(self.bn_list[i](x))

        action = T.tanh(self.mu(x))

        return action