from abc import ABC, abstractmethod
from typing import NamedTuple

from torch import Tensor


class Transition(NamedTuple):
    state: Tensor
    action: Tensor
    next_state: Tensor
    reward: Tensor
    done: Tensor


class BaseExperienceReplay(ABC):
    def __init__(self) -> None:
        self.size: int = 0
        self.batch_size: int

    @abstractmethod
    def push(self, transition: Transition):
        pass

    @abstractmethod
    def sample(self) -> Transition:
        pass

    def __len__(self) -> int:
        return self.size
