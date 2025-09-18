from pathlib import Path
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

    def __init__(
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
        # SAC specific parameters
        tau: float = 0.005,
        gamma: float = 0.99,
        experience_replay_size: int = int(10000),
        batch_size: int = 256,
        target_entropy: float = -1.0,
        learning_starts: int = 1000,
        gradient_steps: int = 1,
    ) -> None:
        if type(network_arch) is str:
            network_arch = literal_eval(network_arch)
        super().__init__(
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
                    "target_entropy": target_entropy,
                    "batch_size": batch_size,
                    "device": device,
                    "normalize_observation": normalize_observation,
                    "gradient_steps": gradient_steps,
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
                env=env, network_arch=network_arch, device=device
            )
        elif network_type == "cnn":
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
        )

    def save(self, folder: str = None, checkpoint="", save_path=None):
        if save_path:
            save_path = Path(save_path).with_suffix(".ckpt")
        else:
            env_name = self.env.spec.id
            folder: Path = Path(folder)
            save_path = (
                folder / f"{env_name}_{self.algo_name}_{self.device}_{checkpoint}"
            )
            save_path = save_path.with_suffix(".ckpt")
        model_state = {
            # Model architecture and metadata
            "state_dict": self.agent.net.state_dict(),
            "network_arch": self.network_arch,
            "network_type": self.network_type,
            "checkpoint": checkpoint,
            "device": self.device,
            "normalize_observation": self.normalize_observation,
            # Optimizer states
            "actor_optimizer": self.agent.actor_optimizer.state_dict(),
            "critic1_optimizer": self.agent.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.agent.critic2_optimizer.state_dict(),
            "alpha_optimizer": self.agent.alpha_optimizer.state_dict(),
            # SAC specific parameters
            "log_alpha": self.agent.log_alpha,
            "steps_done": self.agent.steps_done,
            # Save all hyperparameters for proper reconstruction
            "tau": self.agent.tau,
            "gamma": self.agent.gamma,
            "batch_size": self.agent.batch_size,
            "target_entropy": self.agent.target_entropy,
            "learning_rate": self.agent.actor_optimizer.param_groups[0]["lr"],
            "learning_starts": self.agent.learning_starts,
            "gradient_clipping_max_norm": self.agent.gradient_clipping_max_norm,
        }
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
        if model_path:
            model_path = Path(model_path).with_suffix(".ckpt")
        else:
            env_name = self.env.spec.id
            folder: Path = Path(folder)
            model_path = (
                folder / f"{env_name}_{self.algo_name}_{self.device}_{checkpoint}"
            )
            model_path = model_path.with_suffix(".ckpt")
        loaded_model = torch.load(model_path, map_location=self.device)

        # Only recreate the neural network if needed
        if (
            self.network_arch != loaded_model["network_arch"]
            or self.network_type != loaded_model["network_type"]
        ):
            if loaded_model["network_type"] == "mlp":
                neural_network = make_sac_networks_mlp(
                    env=self.env,
                    network_arch=loaded_model["network_arch"],
                    device=self.device,
                )
            elif loaded_model["network_type"] == "cnn":
                neural_network = make_sac_networks_cnn(env=self.env, device=self.device)

            self.agent.net = neural_network

        # Load the state dictionary
        self.agent.net.load_state_dict(loaded_model["state_dict"])

        # Update agent parameters
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

        # Load temperature parameter
        if "log_alpha" in loaded_model:
            # Create a new tensor on the correct device
            self.agent.log_alpha = torch.tensor(
                loaded_model["log_alpha"].item(),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

        # Skip optimizer loading when in eval mode
        if not eval_mode and all(
            k in loaded_model
            for k in [
                "actor_optimizer",
                "critic1_optimizer",
                "critic2_optimizer",
                "alpha_optimizer",
            ]
        ):
            try:
                self.agent.actor_optimizer.load_state_dict(
                    loaded_model["actor_optimizer"]
                )
                self.agent.critic1_optimizer.load_state_dict(
                    loaded_model["critic1_optimizer"]
                )
                self.agent.critic2_optimizer.load_state_dict(
                    loaded_model["critic2_optimizer"]
                )
                self.agent.alpha_optimizer.load_state_dict(
                    loaded_model["alpha_optimizer"]
                )
            except ValueError as e:
                print(f"Warning: Could not load optimizer states: {e}")

        # Update class variables
        self.network_arch = loaded_model["network_arch"]
        self.network_type = loaded_model["network_type"]
        self.normalize_observation = loaded_model["normalize_observation"]
