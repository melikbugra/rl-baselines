from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from gymnasium import Env
from torch import Tensor

from utils.base_classes.base_experience_replay import BaseExperienceReplay
from utils.base_classes.base_neural_network import BaseNeuralNetwork
from utils.base_classes.base_writer import BaseWriter


class BaseAgent(ABC):
    def __init__(
        self,
        env: Env,
        neural_network: BaseNeuralNetwork,
        experience_replay: BaseExperienceReplay,
        writer: BaseWriter,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.experience_replay: BaseExperienceReplay = experience_replay
        self.writer: BaseWriter = writer

        self.device: str = device

        self.optimizer = optim.AdamW(
            neural_network.parameters(), lr=learning_rate, amsgrad=True
        )

        self.action_type: str = neural_network.action_type
        self.action_num: int = neural_network.action_num

        self.steps_done: int = 0

    @abstractmethod
    def select_action(self, state: Tensor) -> Tensor:
        """Selects an action under exploration strategy

        :param state: Environment state as a tensor
        :type state: Tensor
        :return: Chosen action as a tensor
        :rtype: Tensor
        """
        pass

    def select_random_action(self) -> Tensor:
        """Selects a random action

        :return: Chosen action as a tensor
        :rtype: Tensor
        """
        if self.action_type == "discrete":
            return torch.tensor(
                [[self.env.action_space.sample()]],
                device=self.device,
                dtype=torch.long,
            )
        elif self.action_type == "multidiscrete":
            return torch.tensor(
                np.array([[self.env.action_space.sample()]]),
                device=self.device,
                dtype=torch.long,
            ).squeeze()

    @abstractmethod
    def select_greedy_action(self, state: Tensor) -> Tensor:
        """Selects an action under exploitation strategy

        :param state: Environment state as a tensor
        :type state: Tensor
        :return: Chosen action as a tensor
        :rtype: Tensor
        """
        pass

    @abstractmethod
    def decode_gym_action(self, nn_action_values: Tensor) -> list:
        """Action values as output of a neural network

        :param nn_action: _description_
        :type nn_action: Tensor
        :return: Actions list
        :rtype: list
        """
        pass

    @abstractmethod
    def optimize_model(self, time_step: int) -> None:
        """Optimize the model of the agent"""
        pass

    @abstractmethod
    def get_transitions(self) -> tuple[Tensor]:
        """Returns the transitions"""
        pass

    def compute_loss(
        self,
        state_batch: Tensor,
        next_state_batch: Tensor,
        action_batch: Tensor,
        reward_batch: Tensor,
        mask_batch: Tensor,
    ) -> Tensor:
        """Computes and returns the total loss

        :param state_batch: State batch from Transition namedtuple
        :type state_batch: Tensor
        :param next_state_batch: Next state batch from Transition namedtuple
        :type next_state_batch: Tensor
        :param action_batch: Action batch from Transition namedtuple
        :type action_batch: Tensor
        :param reward_batch: Reward batch from Transition namedtuple
        :type reward_batch: Tensor
        :param mask_batch: Mask batch obtained from the done batch of Transition namedtuple
        :type mask_batch: Tensor
        :param time_step: Training time step
        :type time_step: int
        """
        pass

    def compute_losses(self):
        """In the future this method will be used by GAN based algorithms that have multiple losses"""
        pass

    @abstractmethod
    def update_parameters(self, total_loss: Tensor) -> None:
        """Update the parameters by back propagating

        :param total_loss: Total loss tensor
        :type total_loss: Tensor
        """
        pass
