import torch.nn as nn
import torch.nn.functional as F
import torch as T


class MLP(nn.Module):
    def __init__(self, input_neurons: int, fc_num: int, fc_neuron_nums: list[int], output_neurons: int):
        super().__init__()
        
        if fc_num != len(fc_neuron_nums):
            raise ValueError("Fully connected layer number should be equal to the lenght of list of neuron numbers list.")
        
        self.layer_neuron_nums = [input_neurons] + fc_neuron_nums + [output_neurons]
        
        self.fc_list = nn.ModuleList()

        for i in range(fc_num+1):
            self.fc_list.append(nn.Linear(self.layer_neuron_nums[i], self.layer_neuron_nums[i+1]))

    def forward(self, state):
        x = state
        for i in range(len(self.fc_list) - 1):
            x = F.relu(self.fc_list[i](x))
        actions = self.fc_list[-1](x)

        return actions
    