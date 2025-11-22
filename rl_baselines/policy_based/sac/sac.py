# rl_baselines/policy_based/sac/sac.py

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
                    "target_entropy": target_entropy,
                    "batch_size": batch_size,
                    "device": device,
                    "normalize_observation": normalize_observation,
                    "gradient_steps": gradient_steps,
                    "num_q_heads": num_q_heads,  # <-- eklendi
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
        )

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
