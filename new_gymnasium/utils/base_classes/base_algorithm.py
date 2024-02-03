from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Iterator

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.wrappers import normalize
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.base_classes.base_experience_replay import BaseExperienceReplay, Transition
from utils.base_classes.base_neural_network import BaseNeuralNetwork
from utils.base_classes.base_writer import BaseWriter
from utils.base_classes.base_agent import BaseAgent
from utils.experience_replay.replay_memory import ReplayMemory
from utils.neural_networks.mlp import MLP
from utils.mlflow_logger.mlflow_logger import MLFlowLogger


class BaseAlgorithm(ABC):
    """Base class for RL algorithms"""

    def __init__(
        self,
        env: Env,
        time_steps: int = 100000,
        learning_rate: float = 3e-4,
        network_type: str = "mlp",
        network_arch: list = [128, 128],
        experience_replay_type: str = "er",
        experience_replay_size: int = 10000,
        batch_size: int = 64,
        render: bool = False,
        device: str = "cpu",
        env_seed: int = 42,
        plot_train_sores: bool = False,
        writing_period: int = 500,
        mlflow_tracking_uri: str = None,
        algo_name: str = None,
        normalize_observation: bool = False,
    ) -> None:
        self.env: Env = env
        self.time_steps: int = time_steps
        self.learning_rate: float = learning_rate
        self.network_type: str = network_type
        self.network_arch: list = network_arch
        self.experience_replay_type: str = experience_replay_type
        self.experience_replay_size: int = experience_replay_size
        self.batch_size: int = batch_size
        self.render: bool = render
        self.device: str = device
        self.env_seed: int = env_seed
        self.plot_train_sores: bool = plot_train_sores
        self.writing_period: int = writing_period
        self.normalize_observation: bool = normalize_observation
        if normalize_observation:
            self.env = normalize.NormalizeObservation(env)

        self.algo_name: str

        self.agent: BaseAgent
        self.writer: BaseWriter

        self.train_scores: list[float] = []

        self.mlflow_logger = MLFlowLogger(mlflow_tracking_uri)

        if mlflow_tracking_uri and algo_name:
            self.mlflow_logger.define_experiment_and_run(
                params_to_log={
                    "time_steps": time_steps,
                    "learning_rate": learning_rate,
                    "network_type": network_type,
                    "network_arch": network_arch,
                    "experience_replay_type": experience_replay_type,
                    "experience_replay_size": experience_replay_size,
                    "batch_size": batch_size,
                    "device": device,
                    "normalize_observation": normalize_observation,
                },
                env=env,
                algo_name=algo_name,
            )

    def make_network(self) -> BaseNeuralNetwork:
        """Returns the neural network
        :return: Neural network
        :rtype: BaseNeuralNetwork
        """
        if isinstance(self.env.action_space, Discrete):
            output_neurons = int(self.env.action_space.n)

        elif isinstance(self.env.action_space, MultiDiscrete):
            output_neurons = self.env.action_space.nvec.tolist()

        if self.network_type == "mlp":
            input_neurons = np.prod(self.env.observation_space.shape)

            neural_network = MLP(
                input_neurons=input_neurons,
                network_arch=self.network_arch,
                output_neurons=output_neurons,
            )

        return neural_network

    def make_experience_replay(self) -> BaseExperienceReplay:
        """Returns the experience replay

        :raises NotImplementedError: When the experience replay type is not implemented
        :return: The experience replay
        :rtype: BaseExperienceReplay
        """
        state_dim = np.prod(self.env.observation_space.shape)

        if isinstance(self.env.action_space, Discrete):
            action_dim = self.env.action_space.n

        if isinstance(self.env.action_space, MultiDiscrete):
            action_dim = len(self.env.action_space.nvec)

        if self.experience_replay_type == "er":
            experience_replay = ReplayMemory(
                state_dim=state_dim,
                action_dim=action_dim,
                size=self.experience_replay_size,
                batch_size=self.batch_size,
                device=self.device,
            )
        elif self.experience_replay_type == "per":
            raise NotImplementedError("PER is not implemented yet!")

        return experience_replay

    def train(self):
        """Train the agent"""
        for time_step, transition in self.collect_data():
            self.agent.experience_replay.push(transition)
            self.agent.optimize_model()
            if time_step % self.writing_period == 0 and time_step != 0:
                self.evaluate(time_step)
                print(self.writer)
                self.writer.reset(time_step + self.writing_period)

        self.mlflow_logger.end_run()

        if self.plot_train_sores:
            self.plot_scores(show_result=True)
            plt.show()

    def evaluate(self, time_step: int):
        eval_env: Env = gym.make(self.env.spec.id)
        if self.normalize_observation:
            eval_env = normalize.NormalizeObservation(eval_env)
        state, _ = eval_env.reset(seed=self.env_seed)
        state = self.state_to_torch(state)

        episode_score = 0

        done = False
        while not done:
            action = self.agent.select_greedy_action(state)
            if self.agent.action_type == "discrete":
                action = action.item()
            observation, reward, terminated, truncated, _ = eval_env.step(action)
            episode_score += reward
            reward = torch.tensor([reward], device=self.device)

            if terminated:
                next_state = None
            else:
                next_state = self.state_to_torch(observation)

            done = terminated or truncated

            state = next_state

        self.writer.eval_score = episode_score
        self.mlflow_logger.log_metric(
            "Evaluation Score",
            episode_score,
            step=time_step,
        )

    def collect_data(self) -> Iterator[Transition]:
        """Collect data for training and yield for each time_step

        :yield: The transition for the time_step
        :rtype: Iterator[iter[Transition]]
        """
        state, _ = self.env.reset(seed=self.env_seed)
        state = self.state_to_torch(state)

        episode_score = 0

        for time_step in range(self.time_steps):
            if self.render:
                self.env.render()
            action = self.agent.select_action(state)
            if self.agent.action_type == "discrete":
                action = action.item()
            observation, reward, terminated, truncated, _ = self.env.step(action)
            episode_score += reward
            reward = torch.tensor([reward], device=self.device)

            if terminated:
                next_state = None
            else:
                next_state = self.state_to_torch(observation)

            done = terminated or truncated

            # Store the transition in memory
            transition = Transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
            )
            yield time_step, transition

            state = next_state

            if done:
                self.writer.train_scores.append(episode_score)
                self.train_scores.append(episode_score)
                self.mlflow_logger.log_metric(
                    "Train Score",
                    episode_score,
                    step=time_step,
                )

                if self.plot_train_sores:
                    self.plot_scores()

                state, _ = self.env.reset(seed=self.env_seed)
                state = self.state_to_torch(state)

                episode_score = 0

    def state_to_torch(self, state: np.ndarray):
        return (
            torch.tensor(state, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .view(1, -1)
        )

    def plot_scores(self, show_result=False) -> None:
        """Plot scores with an running average

        :param show_result: Is this the result plot?, defaults to False
        :type show_result: bool, optional
        """
        plt.figure(f"{self.algo_name}")
        scores = torch.tensor(self.train_scores, dtype=torch.float)
        if show_result:
            plt.title(f"{self.algo_name} Result")
        else:
            plt.clf()
            plt.title(f"Training {self.algo_name}...")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.plot(scores.numpy(), color="blue")
        # Take 20 episode averages and plot them too
        if len(scores) >= 20:
            means = scores.unfold(0, 20, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(19), means))
            plt.plot(means.numpy(), color="red")

        plt.pause(0.001)  # pause a bit so that plots are updated
