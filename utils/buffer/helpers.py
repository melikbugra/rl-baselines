from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import torch

from utils.base_classes.base_experience_replay import BaseExperienceReplay, Transition
from utils.buffer.experience_replay import ExperienceReplay


def make_experience_replay(
    env: Env,
    experience_replay_size: int,
    batch_size: int,
    device: torch.device,
) -> BaseExperienceReplay:
    """Returns the experience replay

    :raises NotImplementedError: When the experience replay type is not implemented
    :return: The experience replay
    :rtype: BaseExperienceReplay
    """
    state_dim = np.prod(env.observation_space.shape)

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
    )

    return experience_replay
