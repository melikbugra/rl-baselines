from pathlib import Path

from gymnasium import Env
import torch

from utils.base_classes import BaseAlgorithm, BaseNeuralNetwork
from utils.neural_networks import MLP, make_mlp

from value_based.dqn.vanilla_dqn.vanilla_dqn_agent import VanillaDQNAgent
from value_based.dqn.dqn_writer import DQNWriter


class VanillaDQN(BaseAlgorithm):
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
    ) -> None:
        self.algo_name = "Vanilla-DQN"
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
        )

        if network_type == "mlp":
            neural_network: MLP = make_mlp(
                env=env, network_arch=network_arch, device=device
            )

        self.agent: VanillaDQNAgent = VanillaDQNAgent(
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
        )

    def save(self, folder: str, checkpoint=""):
        env_name = self.env.spec.id
        folder: Path = Path(folder)
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
        }
        torch.save(model_state, save_path)

    def load(self, model_path: str):
        loaded_model = torch.load(model_path, map_location=self.device)

        network_arch = loaded_model["network_arch"]
        network_type = loaded_model["network_type"]
        normalize_observation = loaded_model["normalize_observation"]
        checkpoint = loaded_model["checkpoint"]
        device = loaded_model["device"]

        self.__init__(
            self.env,
            network_arch=network_arch,
            network_type=network_type,
            normalize_observation=normalize_observation,
        )

        self.agent.policy_net.load_state_dict(loaded_model["state_dict"])
        self.agent.optimizer.load_state_dict(loaded_model["optimizer"])
