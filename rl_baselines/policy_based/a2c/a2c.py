from pathlib import Path

from gymnasium import Env
import torch

from rl_baselines.utils.base_classes import BaseAlgorithm, BaseNeuralNetwork
from rl_baselines.utils.neural_networks import (
    ActorCriticMLP,
    make_actor_critic_mlp,
    ActorCriticCNN,
    make_actor_critic_cnn,
)

from rl_baselines.policy_based.a2c.a2c_agent import A2CAgent
from rl_baselines.policy_based.a2c.a2c_writer import A2CWriter


class A2C(BaseAlgorithm):
    algo_name: str = "A2C"

    def __init__(
        self,
        env: Env,
        eval_env_kwargs: dict = {},
        gamma: float = 0.99,
        time_steps: int = 100000,
        # base algorithm attributes
        learning_rate: float = 3e-4,
        network_type: str = "mlp",
        network_arch: list = [128, 128],
        experience_replay_type: str = "tb",
        render: bool = False,
        device: str = "cpu",
        env_seed: int = 42,
        plot_train_sores: bool = False,
        writing_period: int = 10000,
        mlflow_tracking_uri: str = None,
        normalize_observation: bool = False,
        gradient_clipping_max_norm: float = 1.0,
        log_model: bool = False,
        render_eval: bool = False,
        eval_episodes: int = 10,
        # optional a2c attributes
        n_step: int = 5,
    ) -> None:
        self.algo_name = "A2C"
        super().__init__(
            env=env,
            eval_env_kwargs=eval_env_kwargs,
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
            episodic=False,
            time_steps=time_steps,
            log_model=log_model,
            render_eval=render_eval,
            eval_episodes=eval_episodes,
        )

        if mlflow_tracking_uri and self.algo_name:
            self.mlflow_logger.define_experiment_and_run(
                params_to_log={
                    "time_steps": time_steps,
                    "learning_rate": learning_rate,
                    "network_type": network_type,
                    "network_arch": network_arch,
                    "experience_replay_type": experience_replay_type,
                    "gamma": gamma,
                    "device": device,
                    "normalize_observation": normalize_observation,
                },
                env=env,
                algo_name=self.algo_name,
            )

        self.writer: A2CWriter = A2CWriter(
            writing_period=writing_period,
            time_steps=time_steps,
            mlflow_logger=self.mlflow_logger,
        )

        if network_type == "mlp":
            neural_network: ActorCriticMLP = make_actor_critic_mlp(
                env=env, network_arch=network_arch, device=device
            )
        elif network_type == "cnn":
            neural_network: ActorCriticCNN = make_actor_critic_cnn(
                env=env, device=device
            )

        self.agent: A2CAgent = A2CAgent(
            env=env,
            gamma=gamma,
            time_steps=time_steps,
            experience_replay_type=experience_replay_type,
            neural_network=neural_network,
            writer=self.writer,
            learning_rate=learning_rate,
            device=device,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
            n_step=n_step,
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
        if self.log_model:
            self.mlflow_logger.log_artifact(
                local_path=save_path, artifact_path=self.models_folder
            )

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
            eval_env_kwargs=self.eval_env_kwargs,
        )

        self.agent.net.load_state_dict(loaded_model["state_dict"])
        self.agent.optimizer.load_state_dict(loaded_model["optimizer"])
