import torch.nn as nn
import torch.nn.functional as F
import torch as T


class MLP(nn.Module):
    def __init__(self, input_neurons: int, fc_num: int, fc_neuron_nums: list[int], output_neurons: int):
        super().__init__()
        
        if fc_num != len(fc_neuron_nums):
            raise ValueError("Fully connected layer number should be equal to the lenght of list of neuron numbers list.")
        
        self.layer_neuron_nums = [input_neurons] + fc_neuron_nums + [output_neurons]

        self.feature_layer = nn.Linear(self.layer_neuron_nums[0], self.layer_neuron_nums[1])
        
        self.advantage_layer = nn.ModuleList()

        for i in range(1, fc_num+1):
            self.advantage_layer.append(nn.Linear(self.layer_neuron_nums[i], self.layer_neuron_nums[i+1]))

        self.value_layer = nn.ModuleList()

        for i in range(1, fc_num):
            self.value_layer.append(nn.Linear(self.layer_neuron_nums[i], self.layer_neuron_nums[i+1]))
        self.value_layer.append(nn.Linear(self.layer_neuron_nums[-2], 1))

    def forward(self, state):
        x = state

        feature = self.feature_layer(x)

        for i in range(len(self.value_layer) - 1):
            x = F.relu(self.value_layer[i](feature))
        value = self.value_layer[-1](x)

        for i in range(len(self.advantage_layer) - 1):
            x = F.relu(self.advantage_layer[i](feature))
        advantage = self.advantage_layer[-1](x)

        actions = value + advantage - advantage.mean()

        return actions
    