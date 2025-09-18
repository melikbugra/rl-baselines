from abc import ABC, abstractmethod
import ast
from copy import deepcopy
from pathlib import Path
from typing import Iterator
import time

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.wrappers import NormalizeObservation
import numpy as np
import matplotlib.pyplot as plt
from optuna.trial import BaseTrial
import torch

from rl_baselines.utils.base_classes.base_experience_replay import Transition
from rl_baselines.utils.base_classes.base_writer import BaseWriter
from rl_baselines.utils.base_classes.base_agent import BaseAgent
from rl_baselines.utils.mlflow_logger.mlflow_logger import MLFlowLogger

# from rl_baselines.common.env_wrappers import make_atari_env, make_box2d_viz_env


class BaseAlgorithm(ABC):
    """Base class for RL algorithms"""

    algo_name: str

    def __init__(
        self,
        env: Env,
        eval_env_kwargs: dict = {},
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
        gradient_clipping_max_norm: float = None,
        gradient_clipping_value: float = None,
        render_eval: bool = False,
        eval_env: Env = None,
        evaluation: bool = True,
        episodic: bool = False,
        episodes_to_train: int = 16,
        log_model: bool = False,
    ) -> None:
        self.env: Env = env
        self.eval_env: Env = eval_env
        self.evaluation: bool = evaluation
        self.eval_env_kwargs: dict = eval_env_kwargs
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
            self.env = NormalizeObservation(env)

        self.gradient_clipping_max_norm: float = gradient_clipping_max_norm
        self.gradient_clipping_value: float = gradient_clipping_value
        self.render_eval: bool = render_eval
        self.episodic: bool = episodic
        self.episodes_to_train: int = episodes_to_train
        self.log_model: bool = log_model

        self.algo_name: str

        self.agent: BaseAgent
        self.writer: BaseWriter

        self.train_scores: list[float] = []
        self.eval_scores: list[float] = []

        self.mlflow_logger = MLFlowLogger(mlflow_tracking_uri)

        self.start_time: float
        self.models_folder: Path = Path("./models")

        self.atari_envs: list[str] = [
            "PongNoFrameskip-v4",
            "BowlingNoFrameskip-v4",
            "ALE/MarioBros-v5",
            "TennisNoFrameskip-v4",
            "SkiingNoFrameskip-v4",
        ]

        self.melikbugra_envs: list[str] = [
            "WorldsHardestGame-v0",
        ]

        self.box_2d_viz_envs: list[str] = ["CarRacing-v2", "ContinuousMazeViz-v0"]

    def train(self, trial: BaseTrial = None) -> float:
        """Train the agent"""
        self.start_time = time.perf_counter()
        if self.episodic:
            last_avg_eval_score = self.train_episodes(trial)
        else:
            last_avg_eval_score = self.train_iterations(trial)

        self.time_elapsed = time.perf_counter() - self.start_time
        self.save(folder=self.models_folder, checkpoint="last")
        if self.mlflow_logger.log:
            self.mlflow_logger.end_run()

        if self.plot_train_sores:
            self.plot_scores(show_result=True)
            plt.show()

        return last_avg_eval_score

    def train_iterations(self, trial: BaseTrial = None) -> float:
        last_avg_eval_score = None
        best_avg_eval_score = -np.inf
        for time_step, transition in self.collect_data_iterations():
            self.agent.experience_replay.push(transition)
            self.agent.optimize_model(time_step)
            if (
                time_step % self.writing_period == 0 and time_step != 0
            ) or time_step == self.time_steps - 1:
                if self.evaluation:
                    last_avg_eval_score = self.evaluate(
                        time_step,
                        episodes=1,
                        render=self.render_eval,  # TODO: make episodes a parameter
                        eval_env=self.eval_env,
                    )

                    # For optuna pruning
                    if trial:
                        trial.report(-last_avg_eval_score, time_step)

                else:
                    self.writer.calculate_averages()
                    last_avg_eval_score = self.writer.avg_train_score
                    if trial:
                        trial.report(-last_avg_eval_score, time_step)

                if last_avg_eval_score >= best_avg_eval_score:
                    self.save(folder=self.models_folder, checkpoint="best_avg")
                    best_avg_eval_score = last_avg_eval_score

                self.writer.time_elapsed = time.perf_counter() - self.start_time
                if not trial:
                    print(self.writer)
                self.writer.reset(time_step + self.writing_period)

        return last_avg_eval_score

    def train_episodes(self, trial: BaseTrial = None) -> float:
        last_avg_eval_score = None
        best_avg_eval_score = -np.inf

        for episode in self.collect_data_episodes():
            self.agent.optimize_model(episode)
            self.episode_finished = False
            if (
                episode % self.writing_period == 0 and episode != 0
            ) or episode == self.episodes_to_train - 1:
                if self.evaluation:
                    last_avg_eval_score = self.evaluate(
                        episode,
                        episodes=5,
                        render=self.render_eval,  # TODO: make episodes a parameter
                    )
                    # For optuna pruning
                    if trial:
                        trial.report(-last_avg_eval_score, episode)

                else:
                    last_avg_eval_score = self.writer.avg_train_score

                    if trial:
                        trial.report(-last_avg_eval_score, episode)

                if last_avg_eval_score >= best_avg_eval_score:
                    self.save(folder=self.models_folder, checkpoint="best_avg")
                    best_avg_eval_score = last_avg_eval_score
                self.writer.time_elapsed = time.perf_counter() - self.start_time
                if not trial:
                    print(self.writer)
                self.writer.reset(episode + self.writing_period)

        return last_avg_eval_score

    def evaluate(
        self,
        time_step: int = None,
        render: bool = True,
        episodes: int = 10,
        print_episode_score: bool = False,
        eval_env: Env = None,
    ):
        self.agent.net.eval()  # Set the model to evaluation mode
        self.agent.net.training = False
        if eval_env is None:
            if self.env.spec.id in self.atari_envs:
                if render:
                    eval_env: Env = make_atari_env(
                        self.env.spec.id, render_mode="human"
                    )
                else:
                    eval_env: Env = make_atari_env(self.env.spec.id)
                if self.normalize_observation:
                    eval_env = NormalizeObservation(eval_env)
            elif self.env.spec.id in self.box_2d_viz_envs:
                if render:
                    eval_env: Env = make_box2d_viz_env(
                        self.env.spec.id,
                        render_mode="human",
                        **self.eval_env_kwargs,
                    )
                else:
                    eval_env: Env = make_box2d_viz_env(
                        self.env.spec.id,
                        **self.eval_env_kwargs,
                    )
                if self.normalize_observation:
                    eval_env = NormalizeObservation(eval_env)
            elif self.env.spec.id in self.melikbugra_envs:
                if render:
                    eval_env: Env = make_atari_env(
                        self.env.spec.id, render_mode="human", fire_reset=False
                    )
                else:
                    eval_env: Env = make_atari_env(self.env.spec.id, fire_reset=False)
                if self.normalize_observation:
                    eval_env = NormalizeObservation(eval_env)
            else:
                if render:
                    eval_env: Env = gym.make(
                        self.env.spec.id,
                        render_mode="human",
                        **self.eval_env_kwargs,
                    )
                else:
                    eval_env: Env = gym.make(
                        self.env.spec.id,
                        **self.eval_env_kwargs,
                    )
                if self.normalize_observation:
                    eval_env = NormalizeObservation(eval_env)

        episode_scores = []
        for _ in range(episodes):
            state, _ = eval_env.reset(seed=np.random.randint(0, 100))
            state = self.state_to_torch(state)

            episode_score = 0

            done = False
            while not done:
                if render:
                    eval_env.render()
                action = self.agent.select_greedy_action(state, eval=True)
                if self.agent.action_type == "discrete":
                    action_to_env = action.item()
                else:
                    action_to_env = action.cpu().numpy().flatten().tolist()
                observation, reward, terminated, truncated, _ = eval_env.step(
                    action_to_env
                )
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
        # if not render:
        self.writer.avg_eval_score = average_score
        self.mlflow_logger.log_metric(
            "Average Evaluation Score",
            average_score,
            step=time_step,
        )

        eval_env.close()
        self.agent.net.train()  # Set the model back to training mode
        self.agent.net.training = True
        return average_score

    def collect_data_iterations(self) -> Iterator[Transition]:
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
                action_to_env = action.item()
            else:
                action_to_env = action.cpu().numpy().flatten().tolist()
            observation, reward, terminated, truncated, _ = self.env.step(action_to_env)
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

    def collect_data_episodes(self) -> Iterator[Transition]:
        """Collect data for training and yield for each time_step

        :yield: The transition for the time_step
        :rtype: Iterator[iter[Transition]]
        """
        state, _ = self.env.reset(seed=self.env_seed)
        state = self.state_to_torch(state)

        episode_score = 0

        for episode in range(self.episodes_to_train):
            done = False
            while True:
                if self.render:
                    self.env.render()
                action = self.agent.select_action(state)
                if self.agent.action_type == "discrete":
                    action_to_env = action.item()
                else:
                    action_to_env = action.cpu().numpy().flatten().tolist()
                observation, reward, terminated, truncated, _ = self.env.step(
                    action_to_env
                )
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

                self.agent.experience_replay.push(transition)

                state = next_state

                if done:
                    self.writer.train_scores.append(episode_score)
                    self.train_scores.append(episode_score)
                    self.mlflow_logger.log_metric(
                        "Train Score",
                        episode_score,
                        step=episode,
                    )

                    if self.plot_train_sores:
                        self.plot_scores()

                    state, _ = self.env.reset(seed=self.env_seed)
                    state = self.state_to_torch(state)

                    episode_score = 0
                    yield episode
                    break

    def state_to_torch(self, state: np.ndarray):
        if self.network_type == "mlp" or self.network_type == "actor_mlp_critic_mlp":
            return torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        elif self.network_type in [
            "cnn",
            "actor_cnn_critic_cnn",
            "actor_critic_cnn",
        ]:
            return torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

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
