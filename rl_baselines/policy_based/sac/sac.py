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

    def save(
        self,
        folder: str = None,
        checkpoint: str = "",
        save_path: str = None,
        save_optimizer: bool = True,
        save_replay_buffer: bool = False,
    ):
        """
        Save model checkpoint.

        Args:
            folder: Directory to save the checkpoint
            checkpoint: Checkpoint name suffix (e.g., 'best', 'last', '100k')
            save_path: Full path to save file (overrides folder/checkpoint)
            save_optimizer: Whether to save optimizer states (needed for resuming training)
            save_replay_buffer: Whether to save replay buffer (large, but allows exact resume)

        Returns:
            Path to saved checkpoint
        """
        # Determine save path
        if save_path:
            save_path = Path(save_path).with_suffix(".ckpt")
        else:
            env_name = (
                self.env.spec.id
                if hasattr(self.env, "spec") and self.env.spec
                else "custom_env"
            )
            folder = Path(folder) if folder else self.models_folder
            folder.mkdir(parents=True, exist_ok=True)
            save_path = (
                folder / f"{env_name}_{self.algo_name}_{self.device}_{checkpoint}"
            ).with_suffix(".ckpt")

        # === Core Model State ===
        model_state = {
            # Version info for compatibility
            "version": "2.0",
            "algo_name": self.algo_name,
            # Network weights (most important for transfer learning)
            "state_dict": self.agent.net.state_dict(),
            # Network architecture (needed to reconstruct)
            "network_arch": self.network_arch,
            "network_type": self.network_type,
            "num_q_heads": len(self.agent.net.critics)
            if hasattr(self.agent.net, "critics")
            else 2,
            # Environment info
            "env_id": self.env.spec.id
            if hasattr(self.env, "spec") and self.env.spec
            else None,
            "action_dim": self.agent.action_dim,
            "observation_space_shape": self._get_obs_space_info(),
            # Training state
            "steps_done": self.agent.steps_done,
            "checkpoint": checkpoint,
            "device": str(self.device),
            # SAC hyperparameters
            "hyperparams": {
                "tau": self.agent.tau,
                "gamma": self.agent.gamma,
                "batch_size": self.agent.batch_size,
                "target_entropy": self.agent.target_entropy,
                "learning_rate": self.agent.actor_optimizer.param_groups[0]["lr"],
                "learning_starts": self.agent.learning_starts,
                "gradient_clipping_max_norm": self.agent.gradient_clipping_max_norm,
                "gradient_steps": self.agent.gradient_steps,
            },
            # Temperature parameter
            "log_alpha": self.agent.log_alpha.detach().cpu(),
            # Normalization
            "normalize_observation": self.normalize_observation,
            # HER parameters
            "her": {
                "enabled": self.experience_replay_type == "her",
                "experience_replay_type": self.experience_replay_type,
                "n_sampled_goal": self.n_sampled_goal,
                "goal_selection_strategy": self.goal_selection_strategy,
            },
        }

        # === Optimizer States (optional, for resuming training) ===
        if save_optimizer:
            optimizer_state = {}
            try:
                optimizer_state["actor_optimizer"] = (
                    self.agent.actor_optimizer.state_dict()
                )
                optimizer_state["alpha_optimizer"] = (
                    self.agent.alpha_optimizer.state_dict()
                )

                if hasattr(self.agent, "critic_optimizer"):
                    optimizer_state["critic_optimizer"] = (
                        self.agent.critic_optimizer.state_dict()
                    )
                elif hasattr(self.agent, "critic1_optimizer"):
                    optimizer_state["critic1_optimizer"] = (
                        self.agent.critic1_optimizer.state_dict()
                    )
                    optimizer_state["critic2_optimizer"] = (
                        self.agent.critic2_optimizer.state_dict()
                    )

                model_state["optimizer_state"] = optimizer_state
            except Exception as e:
                print(f"[SAC.save] Warning: Could not save optimizer states: {e}")

        # === Replay Buffer (optional, large) ===
        if save_replay_buffer:
            try:
                buffer_state = {
                    "size": self.agent.experience_replay.size,
                    "ptr": self.agent.experience_replay.ptr,
                }
                # Only save actual data if buffer has content
                if self.agent.experience_replay.size > 0:
                    size = self.agent.experience_replay.size
                    buffer_state["state_buffer"] = (
                        self.agent.experience_replay.state_buffer[:size].cpu()
                    )
                    buffer_state["action_buffer"] = (
                        self.agent.experience_replay.action_buffer[:size].cpu()
                    )
                    buffer_state["reward_buffer"] = (
                        self.agent.experience_replay.reward_buffer[:size].cpu()
                    )
                    buffer_state["next_state_buffer"] = (
                        self.agent.experience_replay.next_state_buffer[:size].cpu()
                    )
                    buffer_state["done_buffer"] = (
                        self.agent.experience_replay.done_buffer[:size].cpu()
                    )

                    # HER-specific buffers
                    if self.agent.use_her:
                        buffer_state["achieved_goal_buffer"] = (
                            self.agent.experience_replay.achieved_goal_buffer[
                                :size
                            ].cpu()
                        )
                        buffer_state["desired_goal_buffer"] = (
                            self.agent.experience_replay.desired_goal_buffer[
                                :size
                            ].cpu()
                        )
                        buffer_state["next_achieved_goal_buffer"] = (
                            self.agent.experience_replay.next_achieved_goal_buffer[
                                :size
                            ].cpu()
                        )

                model_state["replay_buffer"] = buffer_state
            except Exception as e:
                print(f"[SAC.save] Warning: Could not save replay buffer: {e}")

        # Save to disk
        torch.save(model_state, save_path)
        print(f"[SAC.save] Checkpoint saved to: {save_path}")

        if self.log_model and self.mlflow_logger:
            self.mlflow_logger.log_artifact(
                local_path=str(save_path), artifact_path=str(self.models_folder)
            )

        return save_path

    def _get_obs_space_info(self):
        """Get observation space info for saving."""
        from gymnasium.spaces import Dict as DictSpace

        if isinstance(self.env.observation_space, DictSpace):
            return {
                "type": "goal_env",
                "observation": self.env.observation_space["observation"].shape,
                "achieved_goal": self.env.observation_space["achieved_goal"].shape,
                "desired_goal": self.env.observation_space["desired_goal"].shape,
            }
        else:
            return {
                "type": "standard",
                "shape": self.env.observation_space.shape,
            }

    def load(
        self,
        folder: str = None,
        checkpoint: str = "",
        model_path: str = None,
        load_optimizer: bool = True,
        load_replay_buffer: bool = False,
        load_hyperparams: bool = True,
        strict: bool = False,
        transfer_learning: bool = False,
        eval_mode: bool = None,  # Backward compatibility (deprecated)
    ):
        """
        Load model checkpoint.

        Args:
            folder: Directory containing the checkpoint
            checkpoint: Checkpoint name suffix
            model_path: Full path to checkpoint file (overrides folder/checkpoint)
            load_optimizer: Whether to load optimizer states
            load_replay_buffer: Whether to load replay buffer
            load_hyperparams: Whether to load hyperparameters (tau, gamma, etc.)
            strict: If True, raise error on state_dict mismatch; if False, load what matches
            transfer_learning: If True, only loads network weights, resets training state
            eval_mode: [DEPRECATED] Use load_optimizer=False instead

        Returns:
            Dict with info about what was loaded
        """
        # Handle deprecated eval_mode parameter
        if eval_mode is not None:
            import warnings

            warnings.warn(
                "eval_mode is deprecated. Use load_optimizer=False for evaluation.",
                DeprecationWarning,
            )
            load_optimizer = not eval_mode

        # Determine load path
        if model_path:
            model_path = Path(model_path).with_suffix(".ckpt")
        else:
            env_name = (
                self.env.spec.id
                if hasattr(self.env, "spec") and self.env.spec
                else "custom_env"
            )
            folder = Path(folder) if folder else self.models_folder
            model_path = (
                folder / f"{env_name}_{self.algo_name}_{self.device}_{checkpoint}"
            ).with_suffix(".ckpt")

        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        load_info = {"path": str(model_path), "loaded": [], "warnings": []}

        # === Version Check ===
        version = loaded.get("version", "1.0")
        load_info["version"] = version

        # === Network Architecture Check ===
        ckpt_network_type = loaded.get("network_type", self.network_type)
        ckpt_network_arch = loaded.get("network_arch", self.network_arch)
        ckpt_num_q_heads = loaded.get("num_q_heads", 2)

        current_num_heads = (
            len(self.agent.net.critics) if hasattr(self.agent.net, "critics") else 2
        )

        arch_mismatch = (
            self.network_type != ckpt_network_type
            or self.network_arch != ckpt_network_arch
            or current_num_heads != ckpt_num_q_heads
        )

        if arch_mismatch and not transfer_learning:
            # Rebuild network to match checkpoint
            print(f"[SAC.load] Rebuilding network to match checkpoint architecture")
            if ckpt_network_type == "mlp":
                new_net = make_sac_networks_mlp(
                    env=self.env,
                    network_arch=ckpt_network_arch,
                    device=self.device,
                    num_q_heads=ckpt_num_q_heads,
                )
            else:
                new_net = make_sac_networks_cnn(env=self.env, device=self.device)

            self.agent.net = new_net
            self.network_type = ckpt_network_type
            self.network_arch = ckpt_network_arch
            load_info["loaded"].append("rebuilt_network")

        # === Load Network Weights ===
        state_dict = loaded.get("state_dict", loaded)  # Support old format
        try:
            if strict:
                self.agent.net.load_state_dict(state_dict)
                load_info["loaded"].append("state_dict (strict)")
            else:
                missing, unexpected = self.agent.net.load_state_dict(
                    state_dict, strict=False
                )
                load_info["loaded"].append("state_dict")
                if missing:
                    load_info["warnings"].append(f"Missing keys: {len(missing)}")
                if unexpected:
                    load_info["warnings"].append(f"Unexpected keys: {len(unexpected)}")
        except Exception as e:
            load_info["warnings"].append(f"state_dict load error: {e}")

        # === Transfer Learning Mode ===
        if transfer_learning:
            # Reset training state for new task
            self.agent.steps_done = 0
            if hasattr(self.agent.experience_replay, "clear"):
                self.agent.experience_replay.clear()
            load_info["loaded"].append("transfer_learning_mode")
            print(f"[SAC.load] Transfer learning mode: training state reset")
            return load_info

        # === Load Training State ===
        self.agent.steps_done = loaded.get("steps_done", 0)
        load_info["loaded"].append(f"steps_done={self.agent.steps_done}")

        # === Load Temperature (Alpha) ===
        if "log_alpha" in loaded:
            alpha_val = loaded["log_alpha"]
            if hasattr(alpha_val, "item"):
                alpha_val = alpha_val.item()
            self.agent.log_alpha = torch.tensor(
                alpha_val,
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )
            # Recreate alpha optimizer with new parameter
            lr = self.agent.alpha_optimizer.param_groups[0]["lr"]
            self.agent.alpha_optimizer = torch.optim.Adam([self.agent.log_alpha], lr=lr)
            load_info["loaded"].append("log_alpha")

        # === Load Hyperparameters ===
        if load_hyperparams:
            hyperparams = loaded.get("hyperparams", loaded)  # Support old format

            self.agent.tau = hyperparams.get("tau", self.agent.tau)
            self.agent.gamma = hyperparams.get("gamma", self.agent.gamma)
            self.agent.batch_size = hyperparams.get("batch_size", self.agent.batch_size)
            self.agent.target_entropy = hyperparams.get(
                "target_entropy", self.agent.target_entropy
            )
            self.agent.learning_starts = hyperparams.get(
                "learning_starts", self.agent.learning_starts
            )
            self.agent.gradient_clipping_max_norm = hyperparams.get(
                "gradient_clipping_max_norm", self.agent.gradient_clipping_max_norm
            )
            self.agent.gradient_steps = hyperparams.get(
                "gradient_steps", getattr(self.agent, "gradient_steps", 1)
            )

            self.normalize_observation = loaded.get(
                "normalize_observation", self.normalize_observation
            )
            load_info["loaded"].append("hyperparams")

        # === Load HER Settings ===
        her_config = loaded.get("her", {})
        if her_config.get("enabled", False):
            self.experience_replay_type = her_config.get(
                "experience_replay_type", self.experience_replay_type
            )
            self.n_sampled_goal = her_config.get("n_sampled_goal", self.n_sampled_goal)
            self.goal_selection_strategy = her_config.get(
                "goal_selection_strategy", self.goal_selection_strategy
            )
            load_info["loaded"].append("her_config")

        # === Load Optimizer States ===
        if load_optimizer and "optimizer_state" in loaded:
            try:
                opt_state = loaded["optimizer_state"]

                if "actor_optimizer" in opt_state:
                    self.agent.actor_optimizer.load_state_dict(
                        opt_state["actor_optimizer"]
                    )
                if "alpha_optimizer" in opt_state:
                    self.agent.alpha_optimizer.load_state_dict(
                        opt_state["alpha_optimizer"]
                    )

                if "critic_optimizer" in opt_state and hasattr(
                    self.agent, "critic_optimizer"
                ):
                    self.agent.critic_optimizer.load_state_dict(
                        opt_state["critic_optimizer"]
                    )
                elif "critic1_optimizer" in opt_state and hasattr(
                    self.agent, "critic1_optimizer"
                ):
                    self.agent.critic1_optimizer.load_state_dict(
                        opt_state["critic1_optimizer"]
                    )
                    self.agent.critic2_optimizer.load_state_dict(
                        opt_state["critic2_optimizer"]
                    )

                load_info["loaded"].append("optimizer_state")
            except Exception as e:
                load_info["warnings"].append(f"optimizer_state load error: {e}")
        # Support old format without "optimizer_state" wrapper
        elif load_optimizer and "actor_optimizer" in loaded:
            try:
                self.agent.actor_optimizer.load_state_dict(loaded["actor_optimizer"])
                if "alpha_optimizer" in loaded:
                    self.agent.alpha_optimizer.load_state_dict(
                        loaded["alpha_optimizer"]
                    )
                if "critic_optimizer" in loaded and hasattr(
                    self.agent, "critic_optimizer"
                ):
                    self.agent.critic_optimizer.load_state_dict(
                        loaded["critic_optimizer"]
                    )
                elif "critic1_optimizer" in loaded and hasattr(
                    self.agent, "critic1_optimizer"
                ):
                    self.agent.critic1_optimizer.load_state_dict(
                        loaded["critic1_optimizer"]
                    )
                    self.agent.critic2_optimizer.load_state_dict(
                        loaded["critic2_optimizer"]
                    )
                load_info["loaded"].append("optimizer_state (old format)")
            except Exception as e:
                load_info["warnings"].append(f"optimizer_state load error: {e}")

        # === Load Replay Buffer ===
        if load_replay_buffer and "replay_buffer" in loaded:
            try:
                buf = loaded["replay_buffer"]
                self.agent.experience_replay.size = buf["size"]
                self.agent.experience_replay.ptr = buf["ptr"]

                if buf["size"] > 0:
                    size = buf["size"]
                    self.agent.experience_replay.state_buffer[:size] = buf[
                        "state_buffer"
                    ].to(self.device)
                    self.agent.experience_replay.action_buffer[:size] = buf[
                        "action_buffer"
                    ].to(self.device)
                    self.agent.experience_replay.reward_buffer[:size] = buf[
                        "reward_buffer"
                    ].to(self.device)
                    self.agent.experience_replay.next_state_buffer[:size] = buf[
                        "next_state_buffer"
                    ].to(self.device)
                    self.agent.experience_replay.done_buffer[:size] = buf[
                        "done_buffer"
                    ].to(self.device)

                    # HER-specific buffers
                    if self.agent.use_her and "achieved_goal_buffer" in buf:
                        self.agent.experience_replay.achieved_goal_buffer[:size] = buf[
                            "achieved_goal_buffer"
                        ].to(self.device)
                        self.agent.experience_replay.desired_goal_buffer[:size] = buf[
                            "desired_goal_buffer"
                        ].to(self.device)
                        self.agent.experience_replay.next_achieved_goal_buffer[
                            :size
                        ] = buf["next_achieved_goal_buffer"].to(self.device)

                load_info["loaded"].append(f"replay_buffer (size={buf['size']})")
            except Exception as e:
                load_info["warnings"].append(f"replay_buffer load error: {e}")

        # Print summary
        print(f"[SAC.load] Loaded from: {model_path}")
        print(f"[SAC.load] Components: {', '.join(load_info['loaded'])}")
        if load_info["warnings"]:
            print(f"[SAC.load] Warnings: {', '.join(load_info['warnings'])}")

        return load_info

    @classmethod
    def load_for_eval(cls, model_path: str, env, device: str = "cpu", **kwargs):
        """
        Class method to load a model for evaluation only.

        Args:
            model_path: Path to checkpoint
            env: Environment (can be different from training env for transfer)
            device: Device to load model on
            **kwargs: Additional SAC constructor arguments

        Returns:
            SAC instance ready for evaluation
        """
        loaded = torch.load(model_path, map_location=device, weights_only=False)

        # Extract architecture from checkpoint
        network_arch = loaded.get("network_arch", [256, 256])
        network_type = loaded.get("network_type", "mlp")
        num_q_heads = loaded.get("num_q_heads", 2)

        # Get HER settings
        her_config = loaded.get("her", {})
        experience_replay_type = her_config.get("experience_replay_type", "er")
        n_sampled_goal = her_config.get("n_sampled_goal", 4)
        goal_selection_strategy = her_config.get("goal_selection_strategy", "future")

        # Create SAC instance
        sac = cls(
            env=env,
            network_arch=network_arch,
            network_type=network_type,
            num_q_heads=num_q_heads,
            device=device,
            experience_replay_type=experience_replay_type,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
            evaluation=False,  # Don't need eval env
            **kwargs,
        )

        # Load weights
        sac.load(model_path=model_path, load_optimizer=False, load_replay_buffer=False)

        # Set to eval mode
        sac.agent.net.eval()

        return sac
