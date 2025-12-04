from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict as DictSpace
from typing import Optional, Callable
import numpy as np
import torch

from rl_baselines.utils.base_classes.base_experience_replay import BaseExperienceReplay
from rl_baselines.utils.replay_buffers.experience_replay import ExperienceReplay
from rl_baselines.utils.replay_buffers.prioritized_experience_replay import (
    PrioritizedExperienceReplay,
)
from rl_baselines.utils.replay_buffers.hindsight_experience_replay import (
    HindsightExperienceReplay,
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
    action_type: str = "discrete",
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

    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]

    experience_replay = ExperienceReplay(
        state_dim=state_dim,
        action_dim=action_dim,
        size=experience_replay_size,
        batch_size=batch_size,
        device=device,
        n_step=n_step,
        gamma=gamma,
        action_type=action_type,
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
    action_type: str = "discrete",
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

    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]

    experience_replay = PrioritizedExperienceReplay(
        state_dim=state_dim,
        action_dim=action_dim,
        size=experience_replay_size,
        batch_size=batch_size,
        device=device,
        n_step=n_step,
        gamma=gamma,
        alpha=alpha,
        action_type=action_type,
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


def make_hindsight_experience_replay(
    env: Env,
    experience_replay_size: int,
    batch_size: int,
    device: torch.device,
    n_sampled_goal: int = 4,
    goal_selection_strategy: str = "future",
    gamma: float = 0.99,
    network_type: str = "mlp",
    action_type: str = "continuous",
    compute_reward: Optional[Callable] = None,
) -> HindsightExperienceReplay:
    """
    Creates a Hindsight Experience Replay buffer for goal-conditioned environments.

    Supports both GoalEnv interface (Dict observation space) and standard environments.

    Args:
        env: The gymnasium environment
        experience_replay_size: Maximum buffer size
        batch_size: Batch size for sampling
        device: PyTorch device
        n_sampled_goal: Number of HER goals per transition (default: 4)
        goal_selection_strategy: 'future', 'final', 'episode', or 'random'
        gamma: Discount factor
        network_type: 'mlp' or 'cnn'
        action_type: 'discrete' or 'continuous'
        compute_reward: Custom reward function (optional)

    Returns:
        HindsightExperienceReplay instance
    """
    # Check if environment follows GoalEnv interface
    if hasattr(env, "observation_space") and isinstance(
        env.observation_space, DictSpace
    ):
        # GoalEnv interface
        obs_space = env.observation_space["observation"]
        goal_space = env.observation_space["achieved_goal"]

        if network_type == "mlp":
            state_dim = int(np.prod(obs_space.shape))
            goal_dim = int(np.prod(goal_space.shape))
        else:
            state_dim = obs_space.shape
            goal_dim = goal_space.shape

        # Use environment's compute_reward if available
        if compute_reward is None and hasattr(env, "compute_reward"):
            compute_reward = env.compute_reward
    else:
        # Standard environment - use observation space shape
        if network_type == "mlp":
            state_dim = int(np.prod(env.observation_space.shape))
        else:
            state_dim = env.observation_space.shape
        # For non-GoalEnv, goal_dim defaults to state_dim
        goal_dim = state_dim

    # Action dimension
    if isinstance(env.action_space, Discrete):
        action_dim = 1
    elif isinstance(env.action_space, MultiDiscrete):
        action_dim = len(env.action_space.nvec)
    elif isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = int(np.prod(env.action_space.shape))

    experience_replay = HindsightExperienceReplay(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        size=experience_replay_size,
        batch_size=batch_size,
        device=device,
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy=goal_selection_strategy,
        gamma=gamma,
        action_type=action_type,
        compute_reward=compute_reward,
    )

    return experience_replay
