import numpy as np
import torch
from torch import Tensor

from utils.base_classes.base_experience_replay import BaseExperienceReplay, Transition


class ReplayMemory(BaseExperienceReplay):
    """A simple numpy replay buffer."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        size: int,
        batch_size: int,
        device,
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

    def push(self, transition: Transition):
        self.state_buffer[self.ptr] = transition.state
        # Handle None next_state
        self.next_state_buffer[self.ptr] = (
            transition.next_state
            if transition.next_state is not None
            else torch.zeros_like(self.state_buffer[self.ptr])
        )
        self.action_buffer[self.ptr] = transition.action
        self.reward_buffer[self.ptr] = transition.reward
        self.done_buffer[self.ptr] = transition.done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Transition:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        state = (self.state_buffer[idxs],)
        action = (self.action_buffer[idxs],)
        # Handle None next_state during sampling
        next_state = (torch.stack([self.next_state_buffer[idx] for idx in idxs]),)
        reward = (self.reward_buffer[idxs],)
        done = self.done_buffer[idxs]
        batch = Transition(
            state=state, action=action, next_state=next_state, reward=reward, done=done
        )

        return batch

    def __len__(self) -> int:
        return self.size
