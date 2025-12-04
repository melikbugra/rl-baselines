"""
Hindsight Experience Replay (HER) Implementation

Based on the paper:
"Hindsight Experience Replay" by Andrychowicz et al. (2017)
https://arxiv.org/abs/1707.01495

HER is designed for goal-conditioned reinforcement learning where rewards are sparse.
It works by relabeling failed experiences with alternative goals that were achieved,
allowing the agent to learn from failures.
"""

from collections import deque
from enum import Enum
from typing import Optional, Callable, List, Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor

from rl_baselines.utils.base_classes.base_experience_replay import (
    BaseExperienceReplay,
    Transition,
)


class HERGoalSelectionStrategy(Enum):
    """Goal selection strategies for HER."""

    FUTURE = "future"  # Sample goals from future states in the same episode
    FINAL = "final"  # Use the final state of the episode as goal
    EPISODE = "episode"  # Sample goals from any state in the episode
    RANDOM = "random"  # Sample random achieved goals from the buffer


class GoalConditionedTransition:
    """A transition that includes goal information."""

    __slots__ = [
        "state",
        "action",
        "next_state",
        "reward",
        "done",
        "achieved_goal",
        "desired_goal",
        "next_achieved_goal",
        "info",
    ]

    def __init__(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward: Tensor,
        done: bool,
        achieved_goal: Tensor,
        desired_goal: Tensor,
        next_achieved_goal: Tensor,
        info: Dict[str, Any] = None,
    ):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.achieved_goal = achieved_goal
        self.desired_goal = desired_goal
        self.next_achieved_goal = next_achieved_goal
        self.info = info if info is not None else {}


class HindsightExperienceReplay(BaseExperienceReplay):
    """
    Hindsight Experience Replay (HER) Buffer.

    This replay buffer stores transitions and performs goal relabeling
    using the HER strategy. It's designed for goal-conditioned environments
    that follow the OpenAI Gym GoalEnv interface.

    The environment should provide:
    - observation['observation']: the actual observation
    - observation['achieved_goal']: the goal achieved in current state
    - observation['desired_goal']: the goal we want to achieve
    - compute_reward(achieved_goal, desired_goal, info): reward function

    For environments that don't follow this interface, custom goal extraction
    and reward computation functions can be provided.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        size: int,
        batch_size: int,
        device: torch.device,
        n_sampled_goal: int = 4,
        goal_selection_strategy: str = "future",
        gamma: float = 0.99,
        action_type: str = "continuous",
        compute_reward: Optional[Callable] = None,
        extract_goal_from_state: Optional[Callable] = None,
        handle_timeout_termination: bool = True,
    ):
        """
        Initialize HER buffer.

        Args:
            state_dim: Dimension of the state (observation + goal)
            goal_dim: Dimension of the goal space
            action_dim: Dimension of the action space
            size: Maximum size of the buffer
            batch_size: Size of batches for sampling
            device: PyTorch device
            n_sampled_goal: Number of virtual goals to sample per transition (k in paper)
            goal_selection_strategy: Strategy for selecting goals ('future', 'final', 'episode', 'random')
            gamma: Discount factor (not used in HER directly, but kept for interface compatibility)
            action_type: Type of action space ('discrete' or 'continuous')
            compute_reward: Custom reward computation function.
                           Signature: compute_reward(achieved_goal, desired_goal, info) -> reward
            extract_goal_from_state: Function to extract goal from state if needed.
                                    Signature: extract_goal_from_state(state) -> goal
            handle_timeout_termination: Whether to handle timeout vs true termination differently
        """
        super().__init__()

        self.device = device
        self.state_dim = (
            state_dim if isinstance(state_dim, (list, tuple)) else [state_dim]
        )
        self.goal_dim = goal_dim if isinstance(goal_dim, (list, tuple)) else [goal_dim]
        self.action_dim = action_dim
        self.max_size = size
        self.batch_size = batch_size
        self.gamma = gamma
        self.action_type = action_type
        self.n_sampled_goal = n_sampled_goal
        self.handle_timeout_termination = handle_timeout_termination

        # Goal selection strategy
        if isinstance(goal_selection_strategy, str):
            self.goal_selection_strategy = HERGoalSelectionStrategy(
                goal_selection_strategy.lower()
            )
        else:
            self.goal_selection_strategy = goal_selection_strategy

        # Custom functions
        self._compute_reward = compute_reward
        self._extract_goal_from_state = extract_goal_from_state

        # Main buffer - stores concatenated state+goal
        obs_dim = list(self.state_dim)

        self.state_buffer: Tensor = torch.zeros(
            [size, 1, *obs_dim], dtype=torch.float32, device=device
        )
        self.next_state_buffer: Tensor = torch.zeros(
            [size, 1, *obs_dim], dtype=torch.float32, device=device
        )

        # Goal buffers
        self.achieved_goal_buffer: Tensor = torch.zeros(
            [size, 1, *self.goal_dim], dtype=torch.float32, device=device
        )
        self.next_achieved_goal_buffer: Tensor = torch.zeros(
            [size, 1, *self.goal_dim], dtype=torch.float32, device=device
        )
        self.desired_goal_buffer: Tensor = torch.zeros(
            [size, 1, *self.goal_dim], dtype=torch.float32, device=device
        )

        # Action buffer
        if action_type == "discrete":
            self.action_buffer: Tensor = torch.zeros(
                [size, 1, action_dim], dtype=torch.int64, device=device
            )
        else:
            self.action_buffer: Tensor = torch.zeros(
                [size, 1, action_dim], dtype=torch.float32, device=device
            )

        self.reward_buffer: Tensor = torch.zeros(
            [size, 1, 1], dtype=torch.float32, device=device
        )
        self.done_buffer: Tensor = torch.zeros(
            [size, 1, 1], dtype=torch.bool, device=device
        )

        # Episode tracking for HER
        self.episode_buffer: List[GoalConditionedTransition] = []
        self.episode_start_indices: List[int] = []  # Tracks where each episode starts
        self.current_episode_start: int = 0

        # Buffer pointers
        self.ptr: int = 0
        self.size: int = 0

        # Store achieved goals for 'random' strategy
        self.all_achieved_goals: List[Tensor] = []

    def push(self, transition: GoalConditionedTransition):
        """
        Add a transition to the episode buffer.

        For HER, transitions are first collected in an episode buffer.
        When an episode ends (done=True or timeout), the episode is processed
        and transitions are added to the main buffer with hindsight relabeling.

        Args:
            transition: A GoalConditionedTransition containing state, action, etc.
        """
        self.episode_buffer.append(transition)

        # Store achieved goal for random strategy
        self.all_achieved_goals.append(transition.achieved_goal.clone())
        if len(self.all_achieved_goals) > self.max_size:
            self.all_achieved_goals.pop(0)

        # Check if episode ended
        is_timeout = (
            transition.info.get("TimeLimit.truncated", False)
            if transition.info
            else False
        )
        episode_done = transition.done or is_timeout

        if episode_done:
            self._store_episode()

    def push_simple(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward: Tensor,
        done: bool,
        achieved_goal: Tensor,
        desired_goal: Tensor,
        next_achieved_goal: Tensor,
        info: Dict[str, Any] = None,
    ):
        """
        Convenience method to push a transition without creating GoalConditionedTransition.
        """
        transition = GoalConditionedTransition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            next_achieved_goal=next_achieved_goal,
            info=info,
        )
        self.push(transition)

    def _store_episode(self):
        """
        Process and store an episode with HER relabeling.

        This method:
        1. Stores original transitions
        2. Creates additional transitions with relabeled goals
        3. Clears the episode buffer
        """
        episode_length = len(self.episode_buffer)
        if episode_length == 0:
            return

        # Store original transitions
        for t, transition in enumerate(self.episode_buffer):
            self._store_transition(
                state=transition.state,
                action=transition.action,
                next_state=transition.next_state,
                reward=transition.reward,
                done=transition.done,
                achieved_goal=transition.achieved_goal,
                desired_goal=transition.desired_goal,
                next_achieved_goal=transition.next_achieved_goal,
            )

        # Create HER transitions with relabeled goals
        for t, transition in enumerate(self.episode_buffer):
            # Sample goals based on strategy
            sampled_goals = self._sample_goals(t, episode_length)

            for new_goal in sampled_goals:
                # Compute new reward with the relabeled goal
                new_reward = self.compute_reward(
                    transition.next_achieved_goal,
                    new_goal,
                    transition.info,
                )

                # Check if done should change (goal achieved)
                new_done = self._is_success(transition.next_achieved_goal, new_goal)

                self._store_transition(
                    state=transition.state,
                    action=transition.action,
                    next_state=transition.next_state,
                    reward=new_reward,
                    done=new_done,
                    achieved_goal=transition.achieved_goal,
                    desired_goal=new_goal,
                    next_achieved_goal=transition.next_achieved_goal,
                )

        # Update episode tracking
        self.episode_start_indices.append(self.current_episode_start)
        if (
            len(self.episode_start_indices) > self.max_size // 100
        ):  # Keep reasonable number of episodes
            self.episode_start_indices.pop(0)
        self.current_episode_start = self.ptr

        # Clear episode buffer
        self.episode_buffer.clear()

    def _store_transition(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward: float,
        done: bool,
        achieved_goal: Tensor,
        desired_goal: Tensor,
        next_achieved_goal: Tensor,
    ):
        """Store a single transition in the main buffer."""

        # Ensure tensors are on the right device and have correct shape
        def ensure_tensor(x, dtype=torch.float32):
            if isinstance(x, Tensor):
                return x.to(self.device, dtype=dtype)
            return torch.tensor(x, device=self.device, dtype=dtype)

        state = ensure_tensor(state)
        next_state = ensure_tensor(next_state)
        action = ensure_tensor(
            action, torch.float32 if self.action_type == "continuous" else torch.int64
        )
        achieved_goal = ensure_tensor(achieved_goal)
        desired_goal = ensure_tensor(desired_goal)
        next_achieved_goal = ensure_tensor(next_achieved_goal)

        # Reshape if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if achieved_goal.dim() == 1:
            achieved_goal = achieved_goal.unsqueeze(0)
        if desired_goal.dim() == 1:
            desired_goal = desired_goal.unsqueeze(0)
        if next_achieved_goal.dim() == 1:
            next_achieved_goal = next_achieved_goal.unsqueeze(0)

        self.state_buffer[self.ptr] = state
        self.next_state_buffer[self.ptr] = next_state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = torch.tensor(
            [[reward]], device=self.device, dtype=torch.float32
        )
        self.done_buffer[self.ptr] = torch.tensor(
            [[done]], device=self.device, dtype=torch.bool
        )
        self.achieved_goal_buffer[self.ptr] = achieved_goal
        self.desired_goal_buffer[self.ptr] = desired_goal
        self.next_achieved_goal_buffer[self.ptr] = next_achieved_goal

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _sample_goals(self, transition_idx: int, episode_length: int) -> List[Tensor]:
        """
        Sample goals for HER relabeling based on the selected strategy.

        Args:
            transition_idx: Index of current transition in episode
            episode_length: Total length of the episode

        Returns:
            List of sampled goal tensors
        """
        goals = []

        if self.goal_selection_strategy == HERGoalSelectionStrategy.FINAL:
            # Use the final achieved goal of the episode
            final_goal = self.episode_buffer[-1].next_achieved_goal
            goals = [final_goal.clone() for _ in range(self.n_sampled_goal)]

        elif self.goal_selection_strategy == HERGoalSelectionStrategy.FUTURE:
            # Sample from future states in the episode
            future_indices = list(range(transition_idx + 1, episode_length))
            if len(future_indices) > 0:
                for _ in range(self.n_sampled_goal):
                    future_idx = np.random.choice(future_indices)
                    goals.append(
                        self.episode_buffer[future_idx].next_achieved_goal.clone()
                    )
            else:
                # If no future states, use final state
                final_goal = self.episode_buffer[-1].next_achieved_goal
                goals = [final_goal.clone() for _ in range(self.n_sampled_goal)]

        elif self.goal_selection_strategy == HERGoalSelectionStrategy.EPISODE:
            # Sample from any state in the episode
            for _ in range(self.n_sampled_goal):
                idx = np.random.randint(0, episode_length)
                goals.append(self.episode_buffer[idx].next_achieved_goal.clone())

        elif self.goal_selection_strategy == HERGoalSelectionStrategy.RANDOM:
            # Sample from all achieved goals in the buffer
            if len(self.all_achieved_goals) > 0:
                indices = np.random.choice(
                    len(self.all_achieved_goals),
                    size=min(self.n_sampled_goal, len(self.all_achieved_goals)),
                    replace=True,
                )
                goals = [self.all_achieved_goals[i].clone() for i in indices]
            else:
                # Fallback to final goal
                final_goal = self.episode_buffer[-1].next_achieved_goal
                goals = [final_goal.clone() for _ in range(self.n_sampled_goal)]

        return goals

    def compute_reward(
        self,
        achieved_goal: Tensor,
        desired_goal: Tensor,
        info: Dict[str, Any] = None,
    ) -> float:
        """
        Compute reward for achieving a goal.

        If a custom reward function was provided, use it.
        Otherwise, use sparse reward: 0 if goal achieved, -1 otherwise.

        Args:
            achieved_goal: The goal that was achieved
            desired_goal: The goal we wanted to achieve
            info: Additional info dict

        Returns:
            Computed reward
        """
        if self._compute_reward is not None:
            return self._compute_reward(achieved_goal, desired_goal, info)

        # Default sparse reward
        if self._is_success(achieved_goal, desired_goal):
            return 0.0
        return -1.0

    def _is_success(
        self,
        achieved_goal: Tensor,
        desired_goal: Tensor,
        threshold: float = 0.05,
    ) -> bool:
        """
        Check if the achieved goal matches the desired goal.

        Args:
            achieved_goal: The goal that was achieved
            desired_goal: The goal we wanted to achieve
            threshold: Distance threshold for success

        Returns:
            True if goal was achieved
        """
        # Ensure tensors
        if isinstance(achieved_goal, Tensor):
            achieved = achieved_goal.cpu().numpy().flatten()
        else:
            achieved = np.array(achieved_goal).flatten()

        if isinstance(desired_goal, Tensor):
            desired = desired_goal.cpu().numpy().flatten()
        else:
            desired = np.array(desired_goal).flatten()

        distance = np.linalg.norm(achieved - desired)
        return distance < threshold

    def sample(self) -> Transition:
        """
        Sample a batch of transitions from the buffer.

        Returns concatenated state+goal as the state for the agent.

        Returns:
            Transition tuple with state (observation + goal), action, next_state, reward, done
        """
        assert self.size >= self.batch_size, (
            f"Not enough samples in buffer: {self.size} < {self.batch_size}"
        )

        indices = torch.randperm(self.size, device=self.device)[: self.batch_size]

        # Get base data
        state = self.state_buffer[indices]
        next_state = self.next_state_buffer[indices]
        action = self.action_buffer[indices]
        reward = self.reward_buffer[indices]
        done = self.done_buffer[indices]
        desired_goal = self.desired_goal_buffer[indices]

        # Concatenate state with goal for goal-conditioned policy
        # state_with_goal shape: [batch, 1, state_dim + goal_dim]
        state_with_goal = torch.cat([state, desired_goal], dim=-1)
        next_state_with_goal = torch.cat([next_state, desired_goal], dim=-1)

        return Transition(
            state=state_with_goal,
            action=action,
            next_state=next_state_with_goal,
            reward=reward,
            done=done,
        )

    def sample_with_goals(self) -> Tuple[Transition, Tensor, Tensor, Tensor]:
        """
        Sample a batch with separate goal information.

        Returns:
            Tuple of (Transition, achieved_goal, desired_goal, next_achieved_goal)
        """
        assert self.size >= self.batch_size, (
            f"Not enough samples in buffer: {self.size} < {self.batch_size}"
        )

        indices = torch.randperm(self.size, device=self.device)[: self.batch_size]

        state = self.state_buffer[indices]
        next_state = self.next_state_buffer[indices]
        action = self.action_buffer[indices]
        reward = self.reward_buffer[indices]
        done = self.done_buffer[indices]
        achieved_goal = self.achieved_goal_buffer[indices]
        desired_goal = self.desired_goal_buffer[indices]
        next_achieved_goal = self.next_achieved_goal_buffer[indices]

        transition = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
        )

        return transition, achieved_goal, desired_goal, next_achieved_goal

    def clear(self):
        """Clear all buffers."""
        self.state_buffer = torch.zeros_like(self.state_buffer)
        self.next_state_buffer = torch.zeros_like(self.next_state_buffer)
        self.action_buffer = torch.zeros_like(self.action_buffer)
        self.reward_buffer = torch.zeros_like(self.reward_buffer)
        self.done_buffer = torch.zeros_like(self.done_buffer)
        self.achieved_goal_buffer = torch.zeros_like(self.achieved_goal_buffer)
        self.desired_goal_buffer = torch.zeros_like(self.desired_goal_buffer)
        self.next_achieved_goal_buffer = torch.zeros_like(
            self.next_achieved_goal_buffer
        )

        self.episode_buffer.clear()
        self.episode_start_indices.clear()
        self.all_achieved_goals.clear()

        self.ptr = 0
        self.size = 0
        self.current_episode_start = 0

    def end_episode(self):
        """
        Manually end the current episode and trigger HER processing.

        This is useful when the episode ends due to timeout rather than
        reaching a terminal state.
        """
        if len(self.episode_buffer) > 0:
            self._store_episode()

    def __len__(self) -> int:
        return self.size


def make_hindsight_experience_replay(
    env,
    experience_replay_size: int,
    batch_size: int,
    device: torch.device,
    n_sampled_goal: int = 4,
    goal_selection_strategy: str = "future",
    gamma: float = 0.99,
    action_type: str = "continuous",
    network_type: str = "mlp",
    compute_reward: Optional[Callable] = None,
) -> HindsightExperienceReplay:
    """
    Factory function to create a HER buffer for a goal-conditioned environment.

    Supports environments that follow the OpenAI Gym GoalEnv interface:
    - observation_space is a Dict with 'observation', 'achieved_goal', 'desired_goal'

    Also supports standard environments with custom goal extraction.

    Args:
        env: The gymnasium environment
        experience_replay_size: Maximum buffer size
        batch_size: Batch size for sampling
        device: PyTorch device
        n_sampled_goal: Number of HER goals per transition
        goal_selection_strategy: Goal selection strategy
        gamma: Discount factor
        action_type: 'discrete' or 'continuous'
        network_type: 'mlp' or 'cnn'
        compute_reward: Custom reward function (optional)

    Returns:
        HindsightExperienceReplay instance
    """
    from gymnasium.spaces import Box, Discrete, Dict as DictSpace
    import numpy as np

    # Detect goal-conditioned environment
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
        # Standard environment - use full observation as goal
        if network_type == "mlp":
            state_dim = int(np.prod(env.observation_space.shape))
            goal_dim = state_dim  # Assume goal has same dimension as state
        else:
            state_dim = env.observation_space.shape
            goal_dim = state_dim

    # Action dimension
    if isinstance(env.action_space, Discrete):
        action_dim = 1
    elif isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = int(np.prod(env.action_space.shape))

    return HindsightExperienceReplay(
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
