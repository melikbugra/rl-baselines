from pathlib import Path
from rl_baselines.utils.base_classes.base_algorithm import BaseAlgorithm
from rl_baselines.policy_based.pppppoooooo.ppo_agent import PPOAgent
from rl_baselines.policy_based.pppppoooooo.ppo_writer import PPOWriter
import torch
from rl_baselines.utils.neural_networks import (
    make_actor_critic_mlp,
    make_actor_critic_cnn,
    make_actor_mlp_critic_mlp,
    make_actor_cnn_critic_cnn,
)


class PPO(BaseAlgorithm):
    algo_name: str = "PPO"

    def __init__(
        self,
        env,
        eval_env_kwargs: dict = {},
        time_steps: int = 100000,
        experience_replay_type: str = "tb",
        learning_rate: float = 3e-4,
        network_type: str = "mlp",
        network_arch: list = [128, 128],
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
        clip_range: float = 0.2,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int = 10,
        gamma: float = 0.99,
        memory_size: int = 2048,
        batch_size: int = 64,
    ) -> None:
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
                    "clip_range": clip_range,
                    "gae_lambda": gae_lambda,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "value_coef": value_coef,
                    "entropy_coef": entropy_coef,
                    "memory_size": memory_size,
                    "device": device,
                    "normalize_observation": normalize_observation,
                },
                env=env,
                algo_name=self.algo_name,
            )

        self.writer = PPOWriter(
            writing_period=writing_period,
            time_steps=time_steps,
            mlflow_logger=self.mlflow_logger,
        )

        if network_type == "mlp":
            neural_network = make_actor_critic_mlp(env, network_arch, device)
        elif network_type == "cnn":
            neural_network = make_actor_critic_cnn(env, device)
        elif network_type == "actor_mlp_critic_mlp":
            neural_network = make_actor_mlp_critic_mlp(env, network_arch, device)
        elif network_type == "actor_cnn_critic_cnn":
            neural_network = make_actor_cnn_critic_cnn(env, device)

        self.agent: PPOAgent = PPOAgent(
            env=env,
            writer=self.writer,
            experience_replay_type=experience_replay_type,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
            neural_network=neural_network,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            memory_size=memory_size,
            gamma=gamma,
            n_epochs=n_epochs,
            time_steps=time_steps,
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
