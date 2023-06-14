import torch.nn as nn
import torch.nn.functional as F
import torch
import math



class MLP(nn.Module):
    def __init__(self, input_neurons: int, fc_num: int, fc_neuron_nums: list[int], 
                 output_neurons: int, atom_size: int, support: torch.Tensor):
        super().__init__()
        
        if fc_num != len(fc_neuron_nums):
            raise ValueError("Fully connected layer number should be equal to the lenght of list of neuron numbers list.")
    
        self.atom_size = atom_size
        self.support = support
        self.output_neurons = output_neurons

        self.layer_neuron_nums = [input_neurons] + fc_neuron_nums + [output_neurons*atom_size]

        self.feature_layer = nn.Linear(self.layer_neuron_nums[0], self.layer_neuron_nums[1])

        self.advantage_layer = nn.ModuleList()

        for i in range(1, fc_num+1):
            self.advantage_layer.append(NoisyLinear(self.layer_neuron_nums[i], self.layer_neuron_nums[i+1]))

        self.value_layer = nn.ModuleList()

        for i in range(1, fc_num):
            self.value_layer.append(NoisyLinear(self.layer_neuron_nums[i], self.layer_neuron_nums[i+1]))
        self.value_layer.append(NoisyLinear(self.layer_neuron_nums[-2], atom_size))

    def forward(self, state):
        x = state
        dist = self.dist(x)

        actions = torch.sum(dist * self.support, dim=2)

        return actions

    def dist(self, state: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        x = state
        feature = self.feature_layer(x)

        x = feature
        for i in range(len(self.advantage_layer) - 1):
            x = F.relu(self.advantage_layer[i](x))
        advantage_hidden = x

        x = feature
        for i in range(len(self.value_layer) - 1):
            x = F.relu(self.value_layer[i](x))
        value_hidden = x

        advantage = self.advantage_layer[-1](advantage_hidden).view(
            -1, self.output_neurons, self.atom_size
        )

        value = self.value_layer[-1](value_hidden).view(-1, 1, self.atom_size)

        q_atoms = value + advantage - advantage.mean()
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        for layer in self.advantage_layer:
            layer.reset_noise()
        for layer in self.value_layer:
            layer.reset_noise()


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
    