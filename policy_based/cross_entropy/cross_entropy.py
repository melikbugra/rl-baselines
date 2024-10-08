from pathlib import Path

from gymnasium import Env
import torch

from utils.base_classes import BaseAlgorithm, BaseNeuralNetwork
from utils.neural_networks import MLP, make_mlp, CNN, make_cnn

from policy_based.cross_entropy.cross_entropy_agent import CrossEntropyAgent
from policy_based.cross_entropy.cross_entropy_writer import CrossEntropyWriter


class CrossEntropy(BaseAlgorithm):
    algo_name: str = "Cross-Entropy"

    def __init__(
        self,
        env: Env,
        percentile: int = 70,
        episodes_to_train: int = 16,
        # base algorithm attributes
        time_steps: int = 100000,
        learning_rate: float = 3e-4,
        network_type: str = "mlp",
        network_arch: list = [128, 128],
        experience_replay_type: str = "er",
        experience_replay_size: int = 10000,  # It will be equal to batch size for cross entropy
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
        self.algo_name = "Cross-Entropy"
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

        if self.mlflow_logger.log:
            self.mlflow_logger.log_params(
                {
                    "percentile": percentile,
                }
            )

        self.writer: CrossEntropyWriter = CrossEntropyWriter(
            writing_period=writing_period,
            time_steps=time_steps,
            mlflow_logger=self.mlflow_logger,
        )

        if network_type == "mlp":
            neural_network: MLP = make_mlp(
                env=env, network_arch=network_arch, device=device
            )
        elif network_type == "cnn":
            neural_network: CNN = make_cnn(env=env, device=device)

        self.agent: CrossEntropyAgent = CrossEntropyAgent(
            env=env,
            percentile=percentile,
            episodes_to_train=episodes_to_train,
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
            "state_dict": self.agent.net.state_dict(),
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

        self.agent.net.load_state_dict(loaded_model["state_dict"])
        self.agent.optimizer.load_state_dict(loaded_model["optimizer"])
