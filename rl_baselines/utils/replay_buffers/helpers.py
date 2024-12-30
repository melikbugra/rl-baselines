from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import torch

from rl_baselines.utils.base_classes.base_experience_replay import BaseExperienceReplay
from rl_baselines.utils.replay_buffers.experience_replay import ExperienceReplay
from rl_baselines.utils.replay_buffers.prioritized_experience_replay import (
    PrioritizedExperienceReplay,
)
from rl_baselines.utils.replay_buffers.transition_buffer import TransitionBuffer


def make_experience_replay(
    env: Env,
    experience_replay_size: int,
    batch_size: int,
    device: torch.device,
    n_step: int = 1,
    gamma: float = 0.99,
    network_type: str = "mlp",
) -> BaseExperienceReplay:
    """Returns the experience replay

    :raises NotImplementedError: When the experience replay type is not implemented
    :return: The experience replay
    :rtype: BaseExperienceReplay
    """
    if network_type == "mlp":
        state_dim = np.prod(env.observation_space.shape)
    elif network_type == "cnn":
        state_dim = env.observation_space.shape

    if isinstance(env.action_space, Discrete):
        action_dim = 1

    if isinstance(env.action_space, MultiDiscrete):
        action_dim = len(env.action_space.nvec)

    experience_replay = ExperienceReplay(
        state_dim=state_dim,
        action_dim=action_dim,
        size=experience_replay_size,
        batch_size=batch_size,
        device=device,
        n_step=n_step,
        gamma=gamma,
    )

    return experience_replay


def make_prioritized_experience_replay(
    env: Env,
    experience_replay_size: int,
    batch_size: int,
    device: torch.device,
    n_step: int = 1,
    gamma: float = 0.99,
    network_type: str = "mlp",
    alpha: float = 0.2,
) -> BaseExperienceReplay:
    """Returns the experience replay

    :raises NotImplementedError: When the experience replay type is not implemented
    :return: The experience replay
    :rtype: BaseExperienceReplay
    """
    if network_type == "mlp":
        state_dim = np.prod(env.observation_space.shape)
    elif network_type == "cnn":
        state_dim = env.observation_space.shape

    if isinstance(env.action_space, Discrete):
        action_dim = 1

    if isinstance(env.action_space, MultiDiscrete):
        action_dim = len(env.action_space.nvec)

    experience_replay = PrioritizedExperienceReplay(
        state_dim=state_dim,
        action_dim=action_dim,
        size=experience_replay_size,
        batch_size=batch_size,
        device=device,
        n_step=n_step,
        gamma=gamma,
        alpha=alpha,
    )

    return experience_replay


def make_transition_buffer(
    device: torch.device,
) -> TransitionBuffer:
    """Returns the transition buffer

    :return: The transition buffer
    :rtype: TransitionBuffer
    """

    transition_buffer = TransitionBuffer(
        device=device,
    )

    return transition_buffer
