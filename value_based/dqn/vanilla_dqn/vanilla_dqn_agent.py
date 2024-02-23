from copy import deepcopy

from gymnasium import Env
import numpy as np
import random
from torch import Tensor
import torch

import torch.nn as nn

from utils.base_classes import (
    BaseAgent,
    BaseExperienceReplay,
    BaseNeuralNetwork,
    Transition,
)
from value_based.dqn.dqn_writer import DQNWriter
from utils.buffer import ExperienceReplay, make_experience_replay


class VanillaDQNAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        time_steps: int,
        epsilon_start: float,
        epsilon_end: float,
        exploration_percentage: float,
        gradient_steps: int,
        target_update_frequency: int,
        gamma: float,
        experience_replay_type: str,
        experience_replay_size: int,
        batch_size: int,
        # base agent attributes
        neural_network: BaseNeuralNetwork,
        writer: DQNWriter,
        learning_rate: float = None,
        device: str = None,
    ) -> None:
        super().__init__(
            env=env,
            neural_network=neural_network,
            writer=writer,
            learning_rate=learning_rate,
            device=device,
        )
        self.writer: DQNWriter = writer

        self.epsilon_end: float = epsilon_end
        self.epsilon: float = epsilon_start
        self.epsilon_decay: float = (
            (self.epsilon - self.epsilon_end)
            * 100
            / (time_steps * (exploration_percentage + 1))
        )

        self.gradient_steps: int = gradient_steps
        self.target_update_frequency = target_update_frequency

        self.gamma: float = gamma

        self.policy_net: BaseNeuralNetwork = neural_network
        self.target_net: BaseNeuralNetwork = deepcopy(neural_network)
        self.target_net.eval()

        if experience_replay_type == "er":
            self.experience_replay: ExperienceReplay = make_experience_replay(
                env=env,
                experience_replay_size=experience_replay_size,
                batch_size=batch_size,
                device=device,
                n_step=2,
                gamma=gamma,
            )

    def select_action(self, state: Tensor) -> Tensor:
        self.adaptive_e_greedy()

        sample = random.uniform(0, 1)
        self.steps_done += 1
        if sample > self.epsilon:
            return self.select_greedy_action(state)
        else:
            return self.select_random_action()

    def select_greedy_action(self, state: Tensor) -> Tensor:
        self.policy_net.eval()
        with torch.no_grad():
            if self.policy_net.action_type == "discrete":
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                max_valued_action: Tensor = self.policy_net(state)[0].max(1)[1]
                return max_valued_action.view(1, 1)
            elif self.policy_net.action_type == "multidiscrete":
                return torch.tensor(
                    self.decode_gym_action(self.policy_net(state)),
                    device=self.device,
                    dtype=torch.long,
                )

    def decode_gym_action(self, nn_action_values: Tensor) -> list:
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        decoded_batch_actions = []
        for dim in range(self.action_num):
            max_valued_sub_action: Tensor = nn_action_values[dim].max(1)[1]

            decoded_batch_actions.append(max_valued_sub_action.view(1, 1).item())

        return decoded_batch_actions

    def adaptive_e_greedy(self):
        self.writer.epsilon = self.epsilon
        if self.epsilon > self.epsilon_end + 1e-10:
            self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end

    def optimize_model(self, time_step: int):
        self.policy_net.train()
        if len(self.experience_replay) < self.experience_replay.batch_size:
            return

        for _ in range(self.gradient_steps):
            transitions = self.get_transitions()

            total_loss = self.compute_loss(*transitions)

            self.update_parameters(total_loss, time_step)

    def get_transitions(self):
        sample_tuple = self.experience_replay.sample()
        transitions: Transition = sample_tuple[0]

        state_batch = transitions.state.squeeze(1)
        next_state_batch = transitions.next_state.squeeze(1)
        action_batch = transitions.action.squeeze(1)
        reward_batch = transitions.reward.squeeze(1)
        done_batch = transitions.done.squeeze(1).int()
        mask_batch = 1 - done_batch

        return (
            state_batch,
            next_state_batch,
            action_batch,
            reward_batch,
            mask_batch,
        )

    def compute_loss(
        self,
        state_batch: Tensor,
        next_state_batch: Tensor,
        action_batch: Tensor,
        reward_batch: Tensor,
        mask_batch: Tensor,
    ) -> Tensor:
        total_loss: Tensor = 0

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        for sub_action in range(self.action_num):
            state_action_values = self.policy_net(state_batch)[sub_action].gather(
                1, action_batch.transpose(0, 1)[sub_action].view(-1, 1)
            )
            next_state_values = torch.zeros(
                self.experience_replay.batch_size, device=self.device
            )
            with torch.no_grad():
                next_state_values = self.target_net(next_state_batch)[sub_action].max(
                    1
                )[0] * mask_batch.squeeze(1)
            expected_state_action_values: Tensor = (
                next_state_values * self.gamma
            ) + reward_batch.squeeze(1)

            criterion = nn.SmoothL1Loss()
            total_loss += criterion(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )

        self.writer.losses.append(total_loss.item())

        return total_loss

    def update_parameters(self, total_loss: Tensor, time_step: int):
        self.optimizer.zero_grad()
        total_loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if time_step % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
