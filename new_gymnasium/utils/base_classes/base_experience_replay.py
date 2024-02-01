from typing import NamedTuple

from torch import Tensor


class Transition(NamedTuple):
    state: Tensor
    action: Tensor
    next_state: Tensor
    reward: Tensor
    done: Tensor


class BaseExperienceReplay:
    def __init__(self) -> None:
        self.size: int = 0
        self.batch_size: int

    def push(self, transition: Transition):
        raise NotImplementedError()

    def sample(self) -> Transition:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.size
