from collections import namedtuple, deque
import torch
import numpy as np
from typing import Dict


Transition = namedtuple('Transition',
                        ('state', 'action', 'prob', 'value', 'reward', 'done'))


class ReplayMemory:
    """A simple numpy replay bufferfer."""

    def __init__(self, state_dim: int, action_dim: int, size: int, device, batch_size: int = 32):
        self.state_buffer = torch.zeros([size, 1, state_dim], dtype=torch.float32).to(device)
        self.action_buffer = torch.zeros([size, 1, action_dim], dtype=torch.int64).to(device)
        self.prob_buffer = torch.zeros([size, 1, 1], dtype=torch.float32).to(device)
        self.value_buffer = torch.zeros([size, 1, 1], dtype=torch.float32).to(device)
        self.reward_buffer = torch.zeros([size, 1, 1], dtype=torch.float32).to(device)
        self.done_buffer = torch.zeros([size, 1, 1], dtype=torch.bool).to(device)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def push(self, transition: Transition):
        self.state_buffer[self.ptr] = transition.state
        self.action_buffer[self.ptr] = transition.action
        self.prob_buffer[self.ptr] = transition.prob
        self.value_buffer[self.ptr] = transition.value
        self.reward_buffer[self.ptr] = transition.reward
        self.done_buffer[self.ptr] = transition.done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        batch_start = np.arange(0, self.size, self.batch_size)
        indices = np.arange(self.size, dtype=np.int64)
        np.random.shuffle(indices)
        state=self.state_buffer,
        action=self.action_buffer,
        prob=self.prob_buffer,
        value=self.value_buffer,
        reward=self.reward_buffer,
        done=self.done_buffer
        batch = Transition(state=state,
                            action=action,
                            prob=prob,
                            value=value,
                            reward=reward,
                            done=done)
        mini_batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batch, mini_batches
    
    def clear(self):
        self.state_buffer.zero_()
        self.action_buffer.zero_()
        self.prob_buffer.zero_()
        self.value_buffer.zero_()
        self.reward_buffer.zero_()
        self.done_buffer.zero_()
        self.ptr, self.size, = 0, 0
    
    def __len__(self) -> int:
        return self.size