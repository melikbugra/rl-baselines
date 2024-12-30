from collections import deque

import numpy as np
import torch
from torch import Tensor

from rl_baselines.utils.base_classes.base_experience_replay import (
    Transition,
)


class TransitionBuffer:
    """A transition buffer for storing transitions in an episode."""

    def __init__(
        self,
        device: torch.device,
    ):
        self.device = device

        self.state_buffer: list[Tensor] = []
        self.next_state_buffer: list[Tensor] = []
        self.action_buffer: list[Tensor] = []
        self.reward_buffer: list[Tensor] = []
        self.done_buffer: list[Tensor] = []

    def push(self, transition: Transition):
        self.state_buffer.append(transition.state.clone().detach())
        self.next_state_buffer.append(
            transition.next_state.clone().detach()
            if transition.next_state is not None
            else None
        )
        self.action_buffer.append(transition.action.clone().detach().unsqueeze(0))
        self.reward_buffer.append(transition.reward.clone().detach().unsqueeze(0))
        self.done_buffer.append(
            torch.tensor(transition.done, device=self.device).unsqueeze(0)
        )

    def sample(self) -> Transition:
        state = torch.stack(self.state_buffer)
        next_state = torch.stack(
            [
                ns if ns is not None else torch.zeros_like(self.state_buffer[0])
                for ns in self.next_state_buffer
            ]
        )
        action = torch.stack(self.action_buffer)
        reward = torch.stack(self.reward_buffer)
        done = torch.stack(self.done_buffer)

        batch = Transition(
            state=state, action=action, next_state=next_state, reward=reward, done=done
        )

        return batch

    def clear(self):
        self.state_buffer = []
        self.next_state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
