import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_neurons: int, fc_neuron_nums: list[int]):
        super().__init__()

        output_neurons = 1
               
        self.layer_neuron_nums = [input_neurons] + fc_neuron_nums + [output_neurons]
        
        self.fc_list = nn.ModuleList()

        for i in range(len(fc_neuron_nums)+1):
            self.fc_list.append(nn.Linear(self.layer_neuron_nums[i], self.layer_neuron_nums[i+1]))

    def forward(self, state):
        x = state
        for i in range(len(self.fc_list) - 1):
            x = F.relu(self.fc_list[i](x))
        value = self.fc_list[-1](x)

        return value
    