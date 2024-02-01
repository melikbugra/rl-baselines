from gymnasium import Env

from utils.base_classes import BaseAlgorithm, BaseNeuralNetwork
from utils.experience_replay import ReplayMemory

from value_based.dqn import VanillaDQNAgent


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
        episodes: int = None,
        learning_rate: float = None,
        network_type: str = None,
        network_arch: list = None,
        experience_replay_size: int = None,
        batch_size: int = None,
        render: bool = None,
        device: str = None,
    ) -> None:
        super.__init__(
            env=env,
            episodes=episodes,
            learning_rate=learning_rate,
            network_type=network_type,
            network_arch=network_arch,
            experience_replay_size=experience_replay_size,
            batch_size=batch_size,
            render=render,
            device=device,
        )

        neural_network: BaseNeuralNetwork = self.make_network()

        experience_replay: ReplayMemory = self.make_experience_replay()

        self.agent = VanillaDQNAgent(
            env=env,
            episodes=episodes,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            exploration_percentage=exploration_percentage,
            gamma=gamma,
            neural_network=neural_network,
            experience_replay=experience_replay,
            learning_rate=learning_rate,
            device=device,
        )
