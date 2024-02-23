from collections import deque

import numpy as np
import torch
from torch import Tensor

from utils.base_classes.base_experience_replay import BaseExperienceReplay, Transition


class ExperienceReplay(BaseExperienceReplay):
    """A simple numpy replay buffer."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        size: int,
        batch_size: int,
        device: torch.device,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.state_buffer: Tensor = torch.zeros(
            [size, 1, state_dim], dtype=torch.float32, device=device
        )
        # Use None as a placeholder for next_state (next state is None if the episode is terminated)
        self.next_state_buffer: Tensor = [None] * size
        self.action_buffer: Tensor = torch.zeros(
            [size, 1, action_dim], dtype=torch.int64, device=device
        )
        self.reward_buffer: Tensor = torch.zeros(
            [size, 1, 1], dtype=torch.float32, device=device
        )
        self.done_buffer: Tensor = torch.zeros(
            [size, 1, 1], dtype=torch.bool, device=device
        )
        self.max_size: int = size
        self.batch_size: int = batch_size
        self.ptr: int = 0
        self.size: int = 0

        # for N-step Learning
        self.n_step_buffer: deque[Transition] = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(self, transition: Transition):
        # Add the current transition to the n-step buffer
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate the n-step reward and the final state
        n_step_reward, n_step_state, n_step_done = self._get_n_step_info()

        # Instead of storing None, store a zero tensor for terminal states
        if n_step_state is None:
            n_step_state = torch.zeros_like(self.state_buffer[self.ptr])

        self.state_buffer[self.ptr] = self.n_step_buffer[0].state
        self.next_state_buffer[self.ptr] = n_step_state
        self.action_buffer[self.ptr] = self.n_step_buffer[0].action
        self.reward_buffer[self.ptr] = n_step_reward
        self.done_buffer[self.ptr] = n_step_done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> tuple[Transition, np.ndarray]:
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        state = self.state_buffer[indices]
        action = self.action_buffer[indices]
        next_state = torch.stack([self.next_state_buffer[idx] for idx in indices])
        reward = self.reward_buffer[indices]
        done = self.done_buffer[indices]
        batch = Transition(
            state=state, action=action, next_state=next_state, reward=reward, done=done
        )

        return batch, indices

    def sample_from_indices(self, indices: np.ndarray) -> Transition:
        state = self.state_buffer[indices]
        action = self.action_buffer[indices]
        next_state = torch.stack([self.next_state_buffer[idx] for idx in indices])
        reward = self.reward_buffer[indices]
        done = self.done_buffer[indices]
        batch = Transition(
            state=state, action=action, next_state=next_state, reward=reward, done=done
        )

        return batch, indices

    def clear(self):
        self.state_buffer = torch.zeros_like(self.state_buffer)
        self.next_state_buffer = [None] * self.max_size
        self.action_buffer = torch.zeros_like(self.action_buffer)
        self.reward_buffer = torch.zeros_like(self.reward_buffer)
        self.done_buffer = torch.zeros_like(self.done_buffer)

        self.ptr = 0
        self.size = 0

    def _get_n_step_info(self):
        """Calculate the n-step reward, the final state, action, and done flag."""
        n_step_reward = 0.0
        n_step_state = None
        n_step_action = None
        n_step_done = False

        for i, transition in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma**i) * transition.reward
            n_step_state = transition.next_state
            n_step_done = transition.done

            if transition.done:
                break

        return n_step_reward, n_step_state, n_step_done

    def __len__(self) -> int:
        return self.size
