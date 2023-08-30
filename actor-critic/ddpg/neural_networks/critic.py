import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_neurons: int, fc_neuron_nums: list[int], n_actions: int):
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

        self.action_value = nn.Linear(n_actions, self.layer_neuron_nums[-1])

        fan_in_action = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-fan_in_action, fan_in_action)
        self.action_value.bias.data.uniform_(-fan_in_action, fan_in_action)

        self.q = nn.Linear(self.layer_neuron_nums[-1], 1)

        fan_in_q = 0.003
        self.q.weight.data.uniform_(-fan_in_q, fan_in_q)
        self.q.bias.data.uniform_(-fan_in_q, fan_in_q)

    def forward(self, state, action):
        x = state
        for i in range(len(self.fc_list)-1):
            x = self.fc_list[i](x)
            x = F.relu(self.bn_list[i](x))
        x = self.fc_list[-1](x)
        state_value = self.bn_list[-1](x)

        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value