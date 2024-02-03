from gymnasium import Env

from utils.base_classes import BaseAlgorithm, BaseNeuralNetwork
from utils.experience_replay import ReplayMemory

from value_based.dqn.vanilla_dqn.vanilla_dqn_agent import VanillaDQNAgent
from value_based.dqn.vanilla_dqn.dqn_writer import DQNWriter


class VanillaDQN(BaseAlgorithm):
    def __init__(
        self,
        env: Env,
        epsilon_start: float = 1,
        epsilon_end: float = 0.001,
        exploration_percentage: float = 50,
        gamma: float = 0.99,
        tau: float = 0.005,
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
        writing_period: int = 500,
        mlflow_tracking_uri: str = None,
        normalize_observation: bool = False,
    ) -> None:
        self.algo_name = "Vanilla DQN"
        super().__init__(
            env=env,
            time_steps=time_steps,
            learning_rate=learning_rate,
            network_type=network_type,
            network_arch=network_arch,
            experience_replay_type=experience_replay_type,
            experience_replay_size=experience_replay_size,
            batch_size=batch_size,
            render=render,
            device=device,
            env_seed=env_seed,
            plot_train_sores=plot_train_sores,
            writing_period=writing_period,
            mlflow_tracking_uri=mlflow_tracking_uri,
            algo_name=self.algo_name,
            normalize_observation=normalize_observation,
        )

        self.mlflow_logger.log_params(
            {
                "epsilon_start": epsilon_start,
                "epsilon_end": epsilon_end,
                "exploration_percentage": exploration_percentage,
                "gamma": gamma,
                "tau": tau,
            }
        )

        self.writer: DQNWriter = DQNWriter(
            writing_period=writing_period,
            time_steps=time_steps,
            mlflow_logger=self.mlflow_logger,
        )

        neural_network: BaseNeuralNetwork = self.make_network()

        experience_replay: ReplayMemory = self.make_experience_replay()

        self.agent = VanillaDQNAgent(
            env=env,
            time_steps=time_steps,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            exploration_percentage=exploration_percentage,
            gamma=gamma,
            tau=tau,
            neural_network=neural_network,
            experience_replay=experience_replay,
            writer=self.writer,
            learning_rate=learning_rate,
            device=device,
        )
