import random
from typing import List

import numpy as np
import torch

from utils.replay_buffers.experience_replay import ExperienceReplay, Transition
from utils.replay_buffers.min_segment_tree import MinSegmentTree
from utils.replay_buffers.sum_segment_tree import SumSegmentTree


class PrioritizedExperienceReplay(ExperienceReplay):
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
        alpha: float = 0.2,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            size=size,
            batch_size=batch_size,
            device=device,
            n_step=n_step,
            gamma=gamma,
        )
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def push(self, transition: Transition):
        super().push(transition)

        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, beta: float = 0.4) -> tuple[Transition, np.ndarray, list[int]]:
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        state = self.state_buffer[indices]
        action = self.action_buffer[indices]
        next_state = torch.stack([self.next_state_buffer[idx] for idx in indices])
        reward = self.reward_buffer[indices]
        done = self.done_buffer[indices]
        batch = Transition(
            state=state, action=action, next_state=next_state, reward=reward, done=done
        )

        weights = torch.tensor(
            np.array([self._calculate_weight(i, beta) for i in indices]),
            device=self.device,
        ).view(-1, 1)

        return batch, weights, indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
