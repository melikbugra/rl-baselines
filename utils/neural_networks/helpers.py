from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import torch


from utils.neural_networks.mlp import MLP
from utils.neural_networks.rainbow_mlp import RainbowMLP
from utils.neural_networks.cnn import CNN
from utils.neural_networks.rainbow_cnn import RainbowCNN


def make_mlp(
    env: Env,
    network_arch: list,
    device: torch.device,
) -> MLP:
    """Returns the neural network
    :return: Neural network
    :rtype: MLP
    """
    if isinstance(env.action_space, Discrete):
        output_neurons = int(env.action_space.n)

    elif isinstance(env.action_space, MultiDiscrete):
        output_neurons = env.action_space.nvec.tolist()

    input_neurons = np.prod(env.observation_space.shape)

    neural_network = MLP(
        input_neurons=input_neurons,
        network_arch=network_arch,
        output_neurons=output_neurons,
        device=device,
    )

    return neural_network


def make_cnn(
    env: Env,
    device: torch.device,
) -> CNN:
    if isinstance(env.action_space, Discrete):
        output_neurons = int(env.action_space.n)

    elif isinstance(env.action_space, MultiDiscrete):
        raise Exception("Multidiscrete action is not supported for CNN")

    neural_network = CNN(
        input_shape=env.observation_space.shape,
        output_neurons=output_neurons,
        device=device,
    )

    return neural_network


def make_rainbow_mlp(
    env: Env,
    network_arch: list,
    noisy_enabled: bool = False,
    device: torch.device = "cpu",
) -> RainbowMLP:
    """Returns the neural network
    :return: Neural network
    :rtype: MLP
    """
    if isinstance(env.action_space, Discrete):
        output_neurons = int(env.action_space.n)

    elif isinstance(env.action_space, MultiDiscrete):
        output_neurons = env.action_space.nvec.tolist()

    input_neurons = np.prod(env.observation_space.shape)

    neural_network = RainbowMLP(
        input_neurons=input_neurons,
        network_arch=network_arch,
        output_neurons=output_neurons,
        noisy=noisy_enabled,
        device=device,
    )

    return neural_network


def make_rainbow_cnn(
    env: Env,
    noisy_enabled: bool = False,
    device: torch.device = "cpu",
) -> RainbowCNN:
    if isinstance(env.action_space, Discrete):
        output_neurons = int(env.action_space.n)

    elif isinstance(env.action_space, MultiDiscrete):
        raise Exception("Multidiscrete action is not supported for CNN")

    neural_network = RainbowCNN(
        input_shape=env.observation_space.shape,
        output_neurons=output_neurons,
        noisy=noisy_enabled,
        device=device,
    )

    return neural_network
