from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np

from utils.neural_networks.mlp import MLP
from utils.experience_replay import ReplayMemory
from utils.base_classes import BaseNeuralNetwork, BaseExperienceReplay


class BaseAlgorithm:
    """Base class for RL algorithms"""

    def __init__(
        self,
        env: Env,
        episodes: int = 100,
        learning_rate: float = 3e-4,
        network_type: str = "mlp",
        network_arch: list = [128, 128],
        experience_replay_type: str = "er",
        experience_replay_size: int = 100000,
        batch_size: int = 64,
        render: bool = False,
        device: str = "cpu",
    ) -> None:
        self.env: Env = env
        self.episodes: int = episodes
        self.learning_rate: float = learning_rate
        self.network_type: str = network_type
        self.network_arch: list = network_arch
        self.experience_replay_type: str = experience_replay_type
        self.experience_replay_size: int = experience_replay_size
        self.batch_size: int = batch_size
        self.render: bool = render
        self.device: str = device

    def make_network(self) -> BaseNeuralNetwork:
        """Returns the neural network
        :return: Neural network
        :rtype: BaseNeuralNetwork
        """
        if isinstance(self.env.action_space, Discrete):
            output_neurons = self.env.action_space.n

        elif isinstance(self.env.action_space, MultiDiscrete):
            output_neurons = self.env.action_space.nvec.tolist()

        if self.network_type == "mlp":
            input_neurons = np.prod(self.env.observation_space.shape)

            neural_network = MLP(
                input_neurons=input_neurons,
                network_arch=self.network_arch,
                output_neurons=output_neurons,
            )

        return neural_network

    def make_experience_replay(self) -> BaseExperienceReplay:
        """Returns the experience replay

        :raises NotImplementedError: When the experience replay type is not implemented
        :return: The experience replay
        :rtype: BaseExperienceReplay
        """
        state_dim = np.prod(self.env.observation_space.shape)

        if isinstance(self.env.action_space, Discrete):
            action_dim = self.env.action_space.n

        if isinstance(self.env.action_space, MultiDiscrete):
            action_dim = len(self.env.action_space.nvec)

        if self.experience_replay_type == "er":
            experience_replay = ReplayMemory(
                state_dim=state_dim,
                action_dim=action_dim,
                size=self.experience_replay_size,
                batch_size=self.batch_size,
                device=self.device,
            )
        elif self.experience_replay_type == "per":
            raise NotImplementedError("PER is not implemented yet!")

        return experience_replay
