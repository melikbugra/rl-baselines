from collections import namedtuple, deque
import torch
import numpy as np
from typing import Dict, Deque, Tuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    """A simple numpy replay bufferfer."""

    def __init__(self, state_dim: int, action_dim: int, size: int, batch_size: int = 32, n_step: int = 5, gamma: float = 0.99):
        self.state_buffer = torch.zeros([size, 1, state_dim], dtype=torch.float32)
        self.next_state_buffer = torch.zeros([size, 1, state_dim], dtype=torch.float32)
        self.action_buffer = torch.zeros([size, 1, action_dim], dtype=torch.int64)
        self.reward_buffer = torch.zeros([size, 1, 1], dtype=torch.float32)
        self.done_buffer = torch.zeros([size, 1, 1], dtype=torch.bool)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma


    def push(self, transition: Transition):
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        reward, next_state, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        state = self.n_step_buffer[0].state
        action = self.n_step_buffer[0].action

        self.state_buffer[self.ptr] = state
        self.next_state_buffer[self.ptr] = next_state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        state=self.state_buffer[idxs],
        action=self.action_buffer[idxs],
        next_state=self.next_state_buffer[idxs],
        reward=self.reward_buffer[idxs],
        done=self.done_buffer[idxs]
        batch = Transition(state=state,
                            action=action,
                            next_state=next_state,
                            reward=reward,
                            done=done)
        
        return batch, idxs
    
    def sample_batch_from_idxs(
        self, indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        state=self.state_buffer[indices],
        action=self.action_buffer[indices],
        next_state=self.next_state_buffer[indices],
        reward=self.reward_buffer[indices],
        done=self.done_buffer[indices]
        batch = Transition(state=state,
                            action=action,
                            next_state=next_state,
                            reward=reward,
                            done=done)

        return batch
        
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        reward = n_step_buffer[-1].reward
        next_state = n_step_buffer[-1].next_state
        done = n_step_buffer[-1].done

        for transition in reversed(list(n_step_buffer)[:-1]):
            r = transition.reward
            n_s = transition.next_state
            d = transition.done

            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    def __len__(self) -> int:
        return self.size