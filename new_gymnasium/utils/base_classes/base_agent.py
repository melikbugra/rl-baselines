from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from gymnasium import Env
from torch import Tensor

from utils.base_classes import BaseExperienceReplay, BaseNeuralNetwork


class BaseAgent:
    def __init__(
        self,
        env: Env,
        neural_network: BaseNeuralNetwork,
        experience_replay: BaseExperienceReplay,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.experience_replay: BaseExperienceReplay = experience_replay
        self.device: str = device

        self.optimizer = optim.AdamW(
            neural_network.parameters(), lr=learning_rate, amsgrad=True
        )

        self.action_type: str = neural_network.action_type
        self.action_num: int = neural_network.action_num

        self.steps_done: int = 0
        self.episode_scores: list[float] = []

        self.models_folder: Path = Path("./models")

    def select_action(self, state: Tensor) -> Tensor:
        """Selects an action under exploration strategy

        :param state: Environment state as a tensor
        :type state: Tensor
        :return: Chosen action as a tensor
        :rtype: Tensor
        """
        raise NotImplementedError()

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

    def select_greedy_action(self, state: Tensor) -> Tensor:
        """Selects an action under exploitation strategy

        :param state: Environment state as a tensor
        :type state: Tensor
        :return: Chosen action as a tensor
        :rtype: Tensor
        """
        raise NotImplementedError()

    def decode_gym_action(self, nn_action_values: Tensor) -> list:
        """Action values as output of a neural network

        :param nn_action: _description_
        :type nn_action: Tensor
        :return: Actions list
        :rtype: list
        """
        raise NotImplementedError()

    def plot_scores(self, algo_name: str, show_result=False) -> None:
        """Plot scores with an running average

        :param algo_name: Algorithm name
        :type algo_name: str
        :param show_result: Is this the result plot?, defaults to False
        :type show_result: bool, optional
        """
        plt.figure(f"{algo_name}")
        scores = torch.tensor(self.episode_scores, dtype=torch.float)
        if show_result:
            plt.title(f"{algo_name} Result")
        else:
            plt.clf()
            plt.title(f"Training {algo_name}...")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.plot(scores.numpy(), color="blue")
        # Take 100 episode averages and plot them too
        if len(scores) >= 100:
            means = scores.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color="red")

        plt.pause(0.001)  # pause a bit so that plots are updated

    def optimize_model(self) -> None:
        """Optimize the model of the agent"""
        raise NotImplementedError()

    def get_transitions(self) -> tuple(Tensor):
        """Returns the transitions"""
        raise NotImplementedError()

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
        """
        raise NotImplementedError()

    def update_parameters(self, total_loss: Tensor) -> None:
        """Update the parameters by back propagating

        :param total_loss: Total loss tensor
        :type total_loss: Tensor
        """

    def save_model(self, model: BaseNeuralNetwork, env_name: str, checkpoint=""):
        save_path = self.models_folder / f"{env_name}_{checkpoint}"
        save_path = save_path.with_suffix(".ckpt")
        torch.save(model.state_dict(), save_path)
