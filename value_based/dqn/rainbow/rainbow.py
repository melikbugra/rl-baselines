from pathlib import Path

from gymnasium import Env
import torch

from utils.base_classes import BaseAlgorithm, BaseNeuralNetwork
from utils.neural_networks import (
    RainbowMLP,
    RainbowCNN,
    make_rainbow_mlp,
    make_rainbow_cnn,
)

from value_based.dqn.rainbow.rainbow_agent import RainbowAgent
from value_based.dqn.dqn_writer import DQNWriter


class Rainbow(BaseAlgorithm):
    algo_name: str = "Vanilla-DQN"

    def __init__(
        self,
        env: Env,
        epsilon_start: float = 1,
        epsilon_end: float = 0.001,
        exploration_percentage: float = 50,
        gradient_steps: int = 1,
        target_update_frequency: int = 10,
        gamma: float = 0.99,
        # base algorithm attributes
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
        writing_period: int = 10000,
        mlflow_tracking_uri: str = None,
        normalize_observation: bool = False,
        gradient_clipping_max_norm: float = 1.0,
        # rainbow
        n_step: int = 3,
        double_enabled: bool = True,
        noisy_enabled: bool = True,
        per_alpha: float = 0.2,
        per_beta: float = 0.6,
    ) -> None:
        self.algo_name = "Rainbow"
        super().__init__(
            env=env,
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
        )

        if mlflow_tracking_uri and self.algo_name:
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
                algo_name=self.algo_name,
            )

        if self.mlflow_logger.log:
            self.mlflow_logger.log_params(
                {
                    "epsilon_start": epsilon_start,
                    "epsilon_end": epsilon_end,
                    "exploration_percentage": exploration_percentage,
                    "gamma": gamma,
                }
            )

        self.writer: DQNWriter = DQNWriter(
            writing_period=writing_period,
            time_steps=time_steps,
            mlflow_logger=self.mlflow_logger,
            noisy_enabled=noisy_enabled,
        )

        if network_type == "mlp":
            neural_network: RainbowMLP = make_rainbow_mlp(
                env=env,
                network_arch=network_arch,
                device=device,
                noisy_enabled=noisy_enabled,
            )
        elif network_type == "cnn":
            neural_network: RainbowCNN = make_rainbow_cnn(
                env=env, device=device, noisy_enabled=noisy_enabled
            )

        self.agent: RainbowAgent = RainbowAgent(
            env=env,
            time_steps=time_steps,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            exploration_percentage=exploration_percentage,
            gradient_steps=gradient_steps,
            target_update_frequency=target_update_frequency,
            gamma=gamma,
            experience_replay_type=experience_replay_type,
            experience_replay_size=experience_replay_size,
            batch_size=batch_size,
            neural_network=neural_network,
            writer=self.writer,
            learning_rate=learning_rate,
            device=device,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
            n_step=n_step,
            double_enabled=double_enabled,
            noisy_enabled=noisy_enabled,
            per_alpha=per_alpha,
            per_beta=per_beta,
        )

    def save(self, folder: str, checkpoint=""):
        env_name = self.env.spec.id
        folder: Path = Path(folder)
        env_name = env_name.replace("/", "_")
        save_path = folder / f"{env_name}_{self.algo_name}_{self.device}_{checkpoint}"
        save_path = save_path.with_suffix(".ckpt")
        model_state = {
            "state_dict": self.agent.policy_net.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "network_arch": self.network_arch,
            "network_type": self.network_type,
            "checkpoint": checkpoint,
            "device": self.device,
            "normalize_observation": self.normalize_observation,
            "n_step": self.agent.n_step,
            "experience_replay_type": self.agent.experience_replay_type,
            "double_enabled": self.agent.double_enabled,
            "noisy_enabled": self.agent.noisy_enabled,
        }
        torch.save(model_state, save_path)

    def load(self, model_path: str):
        loaded_model = torch.load(model_path, map_location=self.device)

        network_arch = loaded_model["network_arch"]
        network_type = loaded_model["network_type"]
        normalize_observation = loaded_model["normalize_observation"]
        checkpoint = loaded_model["checkpoint"]
        device = loaded_model["device"]
        n_step = loaded_model["n_step"]
        experience_replay_type = loaded_model["experience_replay_type"]
        double_enabled = loaded_model["double_enabled"]
        noisy_enabled = loaded_model["noisy_enabled"]

        self.__init__(
            self.env,
            network_arch=network_arch,
            network_type=network_type,
            normalize_observation=normalize_observation,
            n_step=n_step,
            experience_replay_type=experience_replay_type,
            double_enabled=double_enabled,
            noisy_enabled=noisy_enabled,
            # device=device,
        )

        self.agent.policy_net.load_state_dict(loaded_model["state_dict"])
        self.agent.optimizer.load_state_dict(loaded_model["optimizer"])
