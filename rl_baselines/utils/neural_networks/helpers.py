from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete, Box
import numpy as np
import torch


from rl_baselines.utils.neural_networks.mlp import MLP
from rl_baselines.utils.neural_networks.rainbow_mlp import RainbowMLP
from rl_baselines.utils.neural_networks.cnn import CNN
from rl_baselines.utils.neural_networks.rainbow_cnn import RainbowCNN
from rl_baselines.utils.neural_networks.actor_critic_mlp import ActorCriticMLP
from rl_baselines.utils.neural_networks.actor_critic_cnn import ActorCriticCNN
from rl_baselines.utils.neural_networks.actor_mlp_critic_mlp import ActorMLPCriticMLP
from rl_baselines.utils.neural_networks.actor_cnn_critic_cnn import ActorCNNCriticCNN


def make_mlp(
    env: Env,
    network_arch: list,
    device: torch.device,
) -> MLP:
    """Returns the neural network
    :return: Neural network
    :rtype: MLP
    """
    input_neurons = np.prod(env.observation_space.shape)

    if isinstance(env.action_space, Discrete):
        output_neurons = int(env.action_space.n)

    elif isinstance(env.action_space, MultiDiscrete):
        output_neurons = env.action_space.nvec.tolist()

    elif isinstance(env.action_space, Box):
        output_neurons = env.action_space.shape

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

    else:
        raise Exception("Action space is not supported")

    neural_network = RainbowCNN(
        input_shape=env.observation_space.shape,
        output_neurons=output_neurons,
        noisy=noisy_enabled,
        device=device,
    )

    return neural_network


def make_actor_critic_mlp(
    env: Env,
    network_arch: list,
    device: torch.device,
) -> ActorCriticMLP:
    """Returns the neural network
    :return: Neural network
    :rtype: MLP
    """
    input_neurons = np.prod(env.observation_space.shape)

    if isinstance(env.action_space, Discrete):
        output_neurons = int(env.action_space.n)

    elif isinstance(env.action_space, MultiDiscrete):
        output_neurons = env.action_space.nvec.tolist()

    elif isinstance(env.action_space, Box):
        output_neurons = env.action_space.shape

    neural_network = ActorCriticMLP(
        input_neurons=input_neurons,
        network_arch=network_arch,
        output_neurons=output_neurons,
        device=device,
    )

    return neural_network


def make_actor_critic_cnn(
    env: Env,
    device: torch.device,
) -> ActorCriticCNN:
    if isinstance(env.action_space, Discrete):
        output_neurons = int(env.action_space.n)

    elif isinstance(env.action_space, MultiDiscrete):
        raise Exception("Multidiscrete action is not supported for CNN")

    neural_network = ActorCriticCNN(
        input_shape=env.observation_space.shape,
        output_neurons=output_neurons,
        device=device,
    )

    return neural_network


def make_actor_mlp_critic_mlp(
    env: Env, network_arch: list, device: torch.device
) -> tuple[MLP, MLP]:
    input_neurons = np.prod(env.observation_space.shape)

    if isinstance(env.action_space, Discrete):
        output_neurons = int(env.action_space.n)

    elif isinstance(env.action_space, MultiDiscrete):
        output_neurons = env.action_space.nvec.tolist()

    elif isinstance(env.action_space, Box):
        output_neurons = env.action_space.shape

    actor = MLP(
        input_neurons=input_neurons,
        network_arch=network_arch,
        output_neurons=output_neurons,
        device=device,
    )

    critic = MLP(
        input_neurons=input_neurons,
        network_arch=network_arch,
        output_neurons=1,
        device=device,
    )

    actor_mlp_critic_mlp = ActorMLPCriticMLP(actor_mlp=actor, critic_mlp=critic)

    return actor_mlp_critic_mlp


def make_actor_cnn_critic_cnn(env: Env, device: torch.device) -> tuple[CNN, CNN]:
    if isinstance(env.action_space, Discrete):
        output_neurons = int(env.action_space.n)

    elif isinstance(env.action_space, MultiDiscrete):
        raise Exception("Multidiscrete action is not supported for CNN")

    actor = CNN(
        input_shape=env.observation_space.shape,
        output_neurons=output_neurons,
        device=device,
    )

    critic = CNN(
        input_shape=env.observation_space.shape,
        output_neurons=1,
        device=device,
    )

    actor_cnn_critic_cnn = ActorCNNCriticCNN(actor_cnn=actor, critic_cnn=critic)

    return actor_cnn_critic_cnn


class MultiCategorical:
    """
    Wrap a list of independent Categorical distributions (one for each discrete action)
    and provide a unified interface.
    """

    def __init__(self, dists):
        self.dists = dists

    def sample(self):
        # Sample from each distribution and stack into one tensor.
        samples = [d.sample() for d in self.dists]
        # Assume each sample has shape (batch_size,); stack along last dim.
        return torch.stack(samples, dim=-1)

    def log_prob(self, actions):
        # Expect actions to be a tensor of shape (..., num_discrete)
        # Compute the log_prob of each component and sum them.
        log_probs = [d.log_prob(actions[..., i]) for i, d in enumerate(self.dists)]
        return sum(log_probs)

    def entropy(self):
        entropies = [d.entropy() for d in self.dists]
        return sum(entropies)
