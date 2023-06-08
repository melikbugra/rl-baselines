import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, input_neurons: int, fc_num: int, fc_neuron_nums: list[int], output_neurons: int, atom_size: int, support: torch.Tensor):
        super().__init__()
        
        if fc_num != len(fc_neuron_nums):
            raise ValueError("Fully connected layer number should be equal to the lenght of list of neuron numbers list.")
        
        self.atom_size = atom_size
        self.support = support
        self.output_neurons = output_neurons
        
        self.layer_neuron_nums = [input_neurons] + fc_neuron_nums + [output_neurons*atom_size] # Noisy
        
        self.fc_list = nn.ModuleList()

        for i in range(fc_num+1):
            self.fc_list.append(nn.Linear(self.layer_neuron_nums[i], self.layer_neuron_nums[i+1]))

    def forward(self, state):
        x = state
        dist = self.dist(x)

        actions = torch.sum(dist * self.support, dim=2)

        return actions
    
    def dist(self, state: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        x = state
        for i in range(len(self.fc_list) - 1):
            x = F.relu(self.fc_list[i](x))
        q_atoms = self.fc_list[-1](x).view(-1, self.output_neurons, self.atom_size)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    