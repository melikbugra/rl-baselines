from abc import ABC, abstractmethod
import ast
from copy import deepcopy
from pathlib import Path
from typing import Iterator
import time

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.wrappers import normalize
import numpy as np
import matplotlib.pyplot as plt
from optuna.trial import BaseTrial
import torch

from utils.base_classes.base_experience_replay import Transition
from utils.base_classes.base_neural_network import BaseNeuralNetwork
from utils.base_classes.base_writer import BaseWriter
from utils.base_classes.base_agent import BaseAgent
from utils.neural_networks.mlp import MLP
from utils.mlflow_logger.mlflow_logger import MLFlowLogger


class BaseAlgorithm(ABC):
    """Base class for RL algorithms"""

    algo_name: str

    def __init__(
        self,
        env: Env,
        time_steps: int = 100000,
        learning_rate: float = 3e-4,
        network_type: str = "mlp",
        network_arch: list | str = [128, 128],
        render: bool = False,
        device: str = "cpu",
        env_seed: int = 42,
        plot_train_sores: bool = False,
        writing_period: int = 10000,
        mlflow_tracking_uri: str = None,
        normalize_observation: bool = False,
    ) -> None:
        self.env: Env = env
        self.time_steps: int = time_steps
        self.learning_rate: float = learning_rate
        self.network_type: str = network_type
        if isinstance(network_arch, str):
            self.network_arch: list = ast.literal_eval(network_arch)
        else:
            self.network_arch: list = network_arch
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

        self.start_time: float
        self.models_folder: Path = Path("./models")

    def train(self, trial: BaseTrial = None) -> float:
        """Train the agent"""
        self.start_time = time.perf_counter()
        best_avg_eval_score = -np.inf
        for time_step, transition in self.collect_data():
            self.agent.experience_replay.push(transition)
            self.agent.optimize_model(time_step)
            if (
                time_step % self.writing_period == 0 and time_step != 0
            ) or time_step == self.time_steps - 1:
                last_avg_eval_score = self.evaluate(time_step, episodes=2, render=False)

                # For optuna pruning
                if trial:
                    trial.report(-last_avg_eval_score, time_step)

                if last_avg_eval_score >= best_avg_eval_score:
                    self.save(folder=self.models_folder, checkpoint="best_avg")
                    best_avg_eval_score = last_avg_eval_score
                self.writer.time_elapsed = time.perf_counter() - self.start_time
                if not trial:
                    print(self.writer)
                self.writer.reset(time_step + self.writing_period)

        self.time_elapsed = time.perf_counter() - self.start_time
        if self.mlflow_logger.log:
            self.mlflow_logger.end_run()

        if self.plot_train_sores:
            self.plot_scores(show_result=True)
            plt.show()

        return last_avg_eval_score

    def evaluate(
        self,
        time_step: int = None,
        render: bool = True,
        episodes: int = 10,
        print_episode_score: bool = False,
    ):
        if render:
            eval_env: Env = gym.make(self.env.spec.id, render_mode="human")
        else:
            eval_env: Env = gym.make(self.env.spec.id)
        if self.normalize_observation:
            eval_env = normalize.NormalizeObservation(eval_env)

        episode_scores = []
        for _ in range(episodes):
            state, _ = eval_env.reset(seed=self.env_seed)
            state = self.state_to_torch(state)

            episode_score = 0

            done = False
            while not done:
                if render:
                    eval_env.render()
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

            episode_scores.append(episode_score)
            if print_episode_score:
                print(f"Score: {episode_score}")

        average_score = np.mean(episode_scores)
        if not render:
            self.writer.avg_eval_score = average_score
            self.mlflow_logger.log_metric(
                "Average Evaluation Score",
                average_score,
                step=time_step,
            )

        return average_score

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

    @abstractmethod
    def save(self, folder: str, checkpoint=""):
        pass

    @abstractmethod
    def load(self):
        pass
