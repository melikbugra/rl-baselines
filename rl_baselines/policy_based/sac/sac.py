# rl_baselines/policy_based/sac/sac.py

from pathlib import Path
from typing import Optional, Callable
import torch
from gymnasium import Env
from ast import literal_eval

from rl_baselines.utils.base_classes.base_algorithm import BaseAlgorithm
from rl_baselines.policy_based.sac.sac_agent import SACAgent
from rl_baselines.policy_based.sac.sac_writer import SACWriter
from rl_baselines.utils.neural_networks.helpers import (
    make_sac_networks_mlp,
    make_sac_networks_cnn,
)


class SAC(BaseAlgorithm):
    algo_name: str = "SAC"

    def __init__(  # ... imza aynı ...
        self,
        env,
        eval_env_kwargs: dict = {},
        time_steps: int = 100000,
        experience_replay_type: str = "er",
        learning_rate: float = 3e-4,
        network_type: str = "mlp",
        network_arch: list | str = [256, 256],
        render: bool = False,
        device: str = "cpu",
        env_seed: int = 42,
        plot_train_sores: bool = False,
        writing_period: int = 10000,
        mlflow_tracking_uri: str = None,
        normalize_observation: bool = False,
        gradient_clipping_max_norm: float = None,
        log_model: bool = False,
        render_eval: bool = False,
        eval_env: Env = None,
        evaluation: bool = True,
        eval_episodes: int = 10,
        # SAC specific parameters
        tau: float = 0.005,
        gamma: float = 0.99,
        experience_replay_size: int = int(10000),
        batch_size: int = 256,
        target_entropy: float = -1.0,
        learning_starts: int = 1000,
        gradient_steps: int = 1,
        num_q_heads: int = 2,
        # HER specific parameters
        n_sampled_goal: int = 4,
        goal_selection_strategy: str = "future",
        her_compute_reward: Optional[Callable] = None,
    ) -> None:
        if type(network_arch) is str:
            network_arch = literal_eval(network_arch)
        super().__init__(  # ... aynı ...
            env=env,
            eval_env_kwargs=eval_env_kwargs,
            time_steps=time_steps,
            learning_rate=learning_rate,
            network_type=network_type,
            network_arch=network_arch,
            render=render,
            device=device,
            env_seed=env_seed,
            plot_train_sores=plot_train_sores,
            writing_period=writing_period,
            mlflow_tracking_uri=mlflow_tracking_uri,
            normalize_observation=normalize_observation,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
            log_model=log_model,
            render_eval=render_eval,
            eval_env=eval_env,
            evaluation=evaluation,
            eval_episodes=eval_episodes,
        )

        if mlflow_tracking_uri and self.algo_name:
            self.mlflow_logger.define_experiment_and_run(
                params_to_log={
                    "time_steps": time_steps,
                    "learning_rate": learning_rate,
                    "network_type": network_type,
                    "network_arch": network_arch,
                    "tau": tau,
                    "gamma": gamma,
                    "experience_replay_size": experience_replay_size,
                    "experience_replay_type": experience_replay_type,
                    "target_entropy": target_entropy,
                    "batch_size": batch_size,
                    "device": device,
                    "normalize_observation": normalize_observation,
                    "gradient_steps": gradient_steps,
                    "num_q_heads": num_q_heads,
                    # HER params
                    "n_sampled_goal": n_sampled_goal
                    if experience_replay_type == "her"
                    else None,
                    "goal_selection_strategy": goal_selection_strategy
                    if experience_replay_type == "her"
                    else None,
                },
                env=env,
                algo_name=self.algo_name,
            )

        self.writer = SACWriter(
            writing_period=writing_period,
            time_steps=time_steps,
            mlflow_logger=self.mlflow_logger,
        )

        if network_type == "mlp":
            neural_network = make_sac_networks_mlp(
                env=env,
                network_arch=network_arch,
                device=device,
                num_q_heads=num_q_heads,  # <-- eklendi
            )
        elif network_type == "cnn":
            # CNN versiyonunda da num_q_heads destekliyorsan buraya geçir.
            neural_network = make_sac_networks_cnn(env=env, device=device)

        self.agent: SACAgent = SACAgent(
            env=env,
            writer=self.writer,
            experience_replay_type=experience_replay_type,
            experience_replay_size=experience_replay_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
            neural_network=neural_network,
            tau=tau,
            gamma=gamma,
            target_entropy=target_entropy,
            learning_starts=learning_starts,
            gradient_steps=gradient_steps,
            # HER specific parameters
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
            her_compute_reward=her_compute_reward,
        )

        # Store HER settings for save/load
        self.experience_replay_type = experience_replay_type
        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy

    def save(self, folder: str = None, checkpoint="", save_path=None):
        if save_path:
            save_path = Path(save_path).with_suffix(".ckpt")
        else:
            env_name = self.env.spec.id
            folder: Path = Path(folder)
            save_path = (
                folder / f"{env_name}_{self.algo_name}_{self.device}_{checkpoint}"
            ).with_suffix(".ckpt")

        # --- ortak metadata ---
        model_state = {
            "state_dict": self.agent.net.state_dict(),
            "network_arch": self.network_arch,
            "network_type": self.network_type,
            "checkpoint": checkpoint,
            "device": self.device,
            "normalize_observation": self.normalize_observation,
            # SAC params
            "log_alpha": self.agent.log_alpha.detach().to("cpu"),
            "steps_done": self.agent.steps_done,
            "tau": self.agent.tau,
            "gamma": self.agent.gamma,
            "batch_size": self.agent.batch_size,
            "target_entropy": self.agent.target_entropy,
            "learning_rate": self.agent.actor_optimizer.param_groups[0]["lr"],
            "learning_starts": self.agent.learning_starts,
            "gradient_clipping_max_norm": self.agent.gradient_clipping_max_norm,
            # Ensemble meta
            "num_q_heads": (
                len(self.agent.net.critics) if hasattr(self.agent.net, "critics") else 2
            ),
            # HER params
            "experience_replay_type": self.experience_replay_type,
            "n_sampled_goal": self.n_sampled_goal,
            "goal_selection_strategy": self.goal_selection_strategy,
        }

        # --- optimizer state'leri (geriye uyumlu) ---
        try:
            # Ensemble tek optimizer
            if hasattr(self.agent, "critic_optimizer"):
                model_state.update(
                    {
                        "actor_optimizer": self.agent.actor_optimizer.state_dict(),
                        "critic_optimizer": self.agent.critic_optimizer.state_dict(),
                        "alpha_optimizer": self.agent.alpha_optimizer.state_dict(),
                    }
                )
            else:
                # Eski iki optimizer
                model_state.update(
                    {
                        "actor_optimizer": self.agent.actor_optimizer.state_dict(),
                        "critic1_optimizer": self.agent.critic1_optimizer.state_dict(),
                        "critic2_optimizer": self.agent.critic2_optimizer.state_dict(),
                        "alpha_optimizer": self.agent.alpha_optimizer.state_dict(),
                    }
                )
        except Exception as e:
            print(f"[SAC.save] Optimizer states not saved: {e}")

        torch.save(model_state, save_path)
        if self.log_model:
            self.mlflow_logger.log_artifact(
                local_path=save_path, artifact_path=self.models_folder
            )

    def load(
        self,
        folder: str = None,
        checkpoint: str = "",
        eval_mode: bool = True,
        model_path=None,
    ):
        from pathlib import Path
        import torch

        if model_path:
            model_path = Path(model_path).with_suffix(".ckpt")
        else:
            env_name = self.env.spec.id
            folder: Path = Path(folder)
            model_path = (
                folder / f"{env_name}_{self.algo_name}_{self.device}_{checkpoint}"
            ).with_suffix(".ckpt")

        loaded_model = torch.load(model_path, map_location=self.device)

        # --- Checkpoint'ten meta ---
        ckpt_network_type = loaded_model.get("network_type", self.network_type)
        ckpt_network_arch = loaded_model.get("network_arch", self.network_arch)
        ckpt_num_q_heads = loaded_model.get("num_q_heads", 2)

        # --- Mevcut ağ ile checkpoint uyum kontrolü ---
        def current_num_heads():
            if hasattr(self.agent.net, "critics"):
                try:
                    return len(self.agent.net.critics)
                except Exception:
                    return 2
            return 2

        needs_rebuild = (
            (self.network_type != ckpt_network_type)
            or (self.network_arch != ckpt_network_arch)
            or (current_num_heads() != ckpt_num_q_heads)
        )

        if needs_rebuild:
            # Ağı checkpoint meta ile yeniden kur
            if ckpt_network_type == "mlp":
                nn_new = make_sac_networks_mlp(
                    env=self.env,
                    network_arch=ckpt_network_arch,
                    device=self.device,
                    num_q_heads=ckpt_num_q_heads,
                )
            else:
                nn_new = make_sac_networks_cnn(
                    env=self.env, device=self.device
                )  # cnn’de de heads varsa ekle
            self.agent.net = nn_new
            # Sınıf alanlarını güncelle
            self.network_type = ckpt_network_type
            self.network_arch = ckpt_network_arch

        # --- Ağı yükle (toleranslı) ---
        missing, unexpected = self.agent.net.load_state_dict(
            loaded_model["state_dict"], strict=False
        )
        if missing or unexpected:
            print(
                f"[SAC.load] state_dict sync -> missing:{len(missing)} unexpected:{len(unexpected)}"
            )

        # --- Agent parametreleri ---
        self.agent.tau = loaded_model.get("tau", self.agent.tau)
        self.agent.gamma = loaded_model.get("gamma", self.agent.gamma)
        self.agent.batch_size = loaded_model.get("batch_size", self.agent.batch_size)
        self.agent.target_entropy = loaded_model.get(
            "target_entropy", self.agent.target_entropy
        )
        self.agent.learning_starts = loaded_model.get(
            "learning_starts", self.agent.learning_starts
        )
        self.agent.gradient_clipping_max_norm = loaded_model.get(
            "gradient_clipping_max_norm", self.agent.gradient_clipping_max_norm
        )
        self.agent.steps_done = loaded_model.get("steps_done", 0)

        # --- Alpha parametresi ---
        if "log_alpha" in loaded_model:
            self.agent.log_alpha = torch.tensor(
                loaded_model["log_alpha"].item(),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

        # --- Optimizer'lar (eval_mode değilse) ---
        if not eval_mode:
            try:
                # Ensemble tek optimizer yolu
                if "critic_optimizer" in loaded_model and hasattr(
                    self.agent, "critic_optimizer"
                ):
                    if "actor_optimizer" in loaded_model:
                        self.agent.actor_optimizer.load_state_dict(
                            loaded_model["actor_optimizer"]
                        )
                    self.agent.critic_optimizer.load_state_dict(
                        loaded_model["critic_optimizer"]
                    )
                    if "alpha_optimizer" in loaded_model:
                        self.agent.alpha_optimizer.load_state_dict(
                            loaded_model["alpha_optimizer"]
                        )
                # Eski iki-optimizer yolu
                elif (
                    "critic1_optimizer" in loaded_model
                    and "critic2_optimizer" in loaded_model
                ):
                    if hasattr(self.agent, "critic1_optimizer") and hasattr(
                        self.agent, "critic2_optimizer"
                    ):
                        if "actor_optimizer" in loaded_model:
                            self.agent.actor_optimizer.load_state_dict(
                                loaded_model["actor_optimizer"]
                            )
                        self.agent.critic1_optimizer.load_state_dict(
                            loaded_model["critic1_optimizer"]
                        )
                        self.agent.critic2_optimizer.load_state_dict(
                            loaded_model["critic2_optimizer"]
                        )
                        if "alpha_optimizer" in loaded_model:
                            self.agent.alpha_optimizer.load_state_dict(
                                loaded_model["alpha_optimizer"]
                            )
            except ValueError as e:
                print(f"[SAC.load] Warning: could not load optimizer states: {e}")

        # Sınıf alanları
        self.normalize_observation = loaded_model.get(
            "normalize_observation", self.normalize_observation
        )

    def train_goal_conditioned(self, trial=None):
        """
        Training loop for goal-conditioned environments with HER support.

        This method should be used when:
        1. experience_replay_type='her'
        2. Environment follows the GoalEnv interface (Dict observation space)

        The environment should provide:
        - observation['observation']: the actual observation
        - observation['achieved_goal']: the goal achieved in current state
        - observation['desired_goal']: the goal we want to achieve
        - compute_reward(achieved_goal, desired_goal, info): reward function

        Returns:
            Last average evaluation score
        """
        import time
        import numpy as np
        from gymnasium.spaces import Dict as DictSpace

        if not self.agent.use_her:
            print("Warning: train_goal_conditioned called but HER is not enabled.")
            print("Falling back to standard training.")
            return self.train()

        # Check if environment follows GoalEnv interface
        is_goal_env = isinstance(self.env.observation_space, DictSpace)
        if not is_goal_env:
            raise ValueError(
                "Environment must follow GoalEnv interface for HER training. "
                "Observation space should be a Dict with 'observation', 'achieved_goal', 'desired_goal'."
            )

        self.start_time = time.perf_counter()
        last_avg_eval_score = None
        best_avg_eval_score = -np.inf

        obs_dict, _ = self.env.reset(seed=self.env_seed)
        state = self._extract_state_with_goal(obs_dict)

        episode_score = 0

        for time_step in range(self.time_steps):
            if self.render:
                self.env.render()

            action = self.agent.select_action(state)

            if self.agent.action_type == "discrete":
                action_to_env = action.item()
            else:
                action_to_env = action.cpu().numpy().flatten().tolist()

            next_obs_dict, reward, terminated, truncated, info = self.env.step(
                action_to_env
            )
            episode_score += reward
            reward_tensor = torch.tensor([reward], device=self.device)

            bootstrap_done = terminated
            episode_done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = self._extract_state_with_goal(next_obs_dict)

            # Extract goals for HER
            achieved_goal = self._to_tensor(obs_dict["achieved_goal"])
            desired_goal = self._to_tensor(obs_dict["desired_goal"])
            next_achieved_goal = self._to_tensor(next_obs_dict["achieved_goal"])
            obs_only = self._to_tensor(obs_dict["observation"])
            next_obs_only = (
                self._to_tensor(next_obs_dict["observation"])
                if not terminated
                else torch.zeros_like(obs_only)
            )

            # Push HER transition
            self.agent.push_her_transition(
                state=obs_only,
                action=action,
                next_state=next_obs_only,
                reward=reward_tensor,
                done=bootstrap_done,
                achieved_goal=achieved_goal,
                desired_goal=desired_goal,
                next_achieved_goal=next_achieved_goal,
                info=info,
            )

            self.agent.optimize_model(time_step)

            state = next_state
            obs_dict = next_obs_dict

            if episode_done:
                # Signal end of episode for HER processing
                self.agent.end_her_episode()

                self.writer.train_scores.append(episode_score)
                self.train_scores.append(episode_score)
                self.mlflow_logger.log_metric(
                    "Train Score",
                    episode_score,
                    step=time_step,
                )

                if self.plot_train_sores:
                    self.plot_scores()

                obs_dict, _ = self.env.reset(seed=self.env_seed)
                state = self._extract_state_with_goal(obs_dict)
                episode_score = 0

            # Evaluation and logging
            if (
                time_step % self.writing_period == 0 and time_step != 0
            ) or time_step == self.time_steps - 1:
                if self.evaluation:
                    last_avg_eval_score = self.evaluate_goal_conditioned(
                        time_step,
                        episodes=self.eval_episodes,
                        render=self.render_eval,
                        eval_env=self.eval_env,
                    )
                    self.save(folder=self.models_folder, checkpoint="last")

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

        self.time_elapsed = time.perf_counter() - self.start_time

        if self.mlflow_logger.log:
            self.mlflow_logger.end_run()

        if self.plot_train_sores:
            self.plot_scores(show_result=True)
            import matplotlib.pyplot as plt

            plt.show()

        return last_avg_eval_score

    def evaluate_goal_conditioned(
        self,
        time_step: int = None,
        render: bool = False,
        episodes: int = 10,
        print_episode_score: bool = False,
        eval_env=None,
    ):
        """
        Evaluation for goal-conditioned environments.

        Args:
            time_step: Current training step
            render: Whether to render
            episodes: Number of evaluation episodes
            print_episode_score: Whether to print each episode score
            eval_env: Optional evaluation environment

        Returns:
            Average evaluation score
        """
        import gymnasium as gym
        import numpy as np
        from gymnasium.spaces import Dict as DictSpace

        self.agent.net.eval()
        self.agent.net.training = False

        if eval_env is None:
            eval_env = gym.make(
                self.env.spec.id,
                render_mode="human" if render else None,
                **self.eval_env_kwargs,
            )

        episode_scores = []
        success_count = 0

        for _ in range(episodes):
            obs_dict, _ = eval_env.reset(seed=self.env_seed)
            state = self._extract_state_with_goal(obs_dict)

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

                next_obs_dict, reward, terminated, truncated, info = eval_env.step(
                    action_to_env
                )
                episode_score += reward

                done = terminated or truncated

                if not done:
                    state = self._extract_state_with_goal(next_obs_dict)
                    obs_dict = next_obs_dict

                # Track success if info provides it
                if info.get("is_success", False):
                    success_count += 1

            episode_scores.append(episode_score)
            if print_episode_score:
                print(f"Score: {episode_score}")

        average_score = np.mean(episode_scores)
        success_rate = success_count / episodes

        self.writer.avg_eval_score = average_score
        self.mlflow_logger.log_metric(
            "Average Evaluation Score",
            average_score,
            step=time_step,
        )
        self.mlflow_logger.log_metric(
            "Success Rate",
            success_rate,
            step=time_step,
        )

        eval_env.close()
        self.agent.net.train()
        self.agent.net.training = True

        return average_score

    def _extract_state_with_goal(self, obs_dict: dict):
        """
        Extract observation and concatenate with goal for goal-conditioned policy.

        Args:
            obs_dict: Dictionary observation from GoalEnv

        Returns:
            Concatenated state+goal tensor
        """
        obs = self._to_tensor(obs_dict["observation"])
        goal = self._to_tensor(obs_dict["desired_goal"])

        # Concatenate observation with goal
        state_with_goal = torch.cat([obs, goal], dim=-1)
        return state_with_goal

    def _to_tensor(self, arr):
        """Convert numpy array to tensor on device."""
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device).float().unsqueeze(0)
        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)
