from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
from utils.neural_networks.mlp import MLP

from utils.base_classes.base_neural_network import BaseNeuralNetwork


def make_mlp(
    env: Env,
    network_type: str,
    network_arch: list,
) -> BaseNeuralNetwork:
    """Returns the neural network
    :return: Neural network
    :rtype: BaseNeuralNetwork
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
    )

    return neural_network
