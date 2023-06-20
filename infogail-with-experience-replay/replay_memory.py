from collections import namedtuple, deque
import torch
import numpy as np
from typing import Dict


Transition = namedtuple('Transition',
                        ('state', 'action'))


class ReplayMemory:
    """A simple numpy replay bufferfer."""

    def __init__(self, state_dim: int, action_dim: int, size: int, batch_size: int = 32):
        self.state_buffer = torch.zeros([size, 1, state_dim], dtype=torch.float32)
        self.action_buffer = torch.zeros([size, 1, action_dim], dtype=torch.int64)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def push(self, transition: Transition):
        self.state_buffer[self.ptr] = transition.state
        self.action_buffer[self.ptr] = transition.action
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        state=self.state_buffer[idxs],
        action=self.action_buffer[idxs],
        batch = Transition(state=state,
                            action=action)
        
        return batch
        

    def __len__(self) -> int:
        return self.size