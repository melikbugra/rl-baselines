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
from rl_baselines.utils.neural_networks.sac_network_mlp import SACNetworkMLP
from rl_baselines.utils.neural_networks.sac_network_cnn import SACNetworkCNN


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

    elif isinstance(env.action_space, Box):
        output_neurons = env.action_space.shape

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

    elif isinstance(env.action_space, Box):
        output_neurons = env.action_space.shape

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


def make_sac_networks_mlp(
    env: Env,
    network_arch,
    device,
    num_q_heads: int = 2,  # <— YENİ: 2 = mevcut davranış
) -> SACNetworkMLP:
    """Creates actor and critic networks for SAC (MLP) with Q-ensemble support"""
    input_neurons = np.prod(env.observation_space.shape)

    assert isinstance(
        env.action_space, Box
    ), "SAC only supports continuous action (Box)"
    act_dim = env.action_space.shape[0]

    # Actor
    actor = MLP(
        input_neurons=input_neurons,
        network_arch=network_arch,
        output_neurons=env.action_space.shape,  # (act_dim,)
        device=device,
    )

    # Critics (heads) — list halinde
    critic_mlps: list[MLP] = []
    target_critic_mlps: list[MLP] = []

    for _ in range(num_q_heads):
        c = MLP(
            input_neurons=input_neurons + act_dim,
            network_arch=network_arch,
            output_neurons=1,
            device=device,
        )
        ct = MLP(
            input_neurons=input_neurons + act_dim,
            network_arch=network_arch,
            output_neurons=1,
            device=device,
        )
        ct.load_state_dict(c.state_dict())
        ct.eval()

        critic_mlps.append(c)
        target_critic_mlps.append(ct)

    sac_network_mlp = SACNetworkMLP(
        actor_mlp=actor,
        critic_mlps=critic_mlps,  # <— liste veriyoruz
        target_critic_mlps=target_critic_mlps,  # <— liste veriyoruz
    )
    return sac_network_mlp


def make_sac_networks_cnn(env: Env, device) -> SACNetworkCNN:
    """Creates actor and critic networks for SAC"""
    input_shape = env.observation_space.shape

    if isinstance(env.action_space, Box):
        actor = CNN(
            input_shape=input_shape,
            output_neurons=env.action_space.shape,
            device=device,
        )
        critic1 = CNN(
            input_shape=input_shape,
            output_neurons=1,
            device=device,
        )
        critic2 = CNN(
            input_shape=input_shape,
            output_neurons=1,
            device=device,
        )
        target_critic1 = CNN(
            input_shape=input_shape,
            output_neurons=1,
            device=device,
        )
        target_critic2 = CNN(
            input_shape=input_shape,
            output_neurons=1,
            device=device,
        )

        target_critic1.load_state_dict(critic1.state_dict())
        target_critic2.load_state_dict(critic2.state_dict())
        target_critic1.eval()
        target_critic2.eval()

        sac_network_cnn = SACNetworkCNN(
            actor_cnn=actor,
            critic1_cnn=critic1,
            critic2_cnn=critic2,
            target_critic1_cnn=target_critic1,
            target_critic2_cnn=target_critic2,
        )

    return sac_network_cnn
