from copy import deepcopy

from gymnasium import Env
import random
from torch import Tensor
import torch

import torch.nn as nn

from utils.base_classes import (
    BaseAgent,
    BaseNeuralNetwork,
    Transition,
)
from value_based.dqn.dqn_writer import DQNWriter
from utils.replay_buffers import (
    ExperienceReplay,
    make_experience_replay,
    PrioritizedExperienceReplay,
    make_prioritized_experience_replay,
)


class RainbowAgent(BaseAgent):
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
        gradient_clipping_max_norm: float = 1.0,
        # rainbow attributes
        n_step: int = 3,
        double_enabled: bool = True,
        noisy_enabled: bool = True,
        per_alpha: float = 0.2,
        per_beta: float = 0.6,
    ) -> None:
        super().__init__(
            env=env,
            neural_network=neural_network,
            writer=writer,
            learning_rate=learning_rate,
            device=device,
        )
        self.writer: DQNWriter = writer

        self.gradient_clipping_max_norm: float = gradient_clipping_max_norm
        self.n_step: int = n_step
        self.double_enabled: bool = double_enabled
        self.noisy_enabled: bool = noisy_enabled
        self.per_enabled: bool = True if experience_replay_type == "per" else False
        self.experience_replay_type = experience_replay_type

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
                n_step=n_step,
                gamma=gamma,
                network_type=self.policy_net.network_type,
            )
        elif experience_replay_type == "per":
            self.per_enabled: bool = True
            self.per_beta: float = per_beta
            self.per_beta_increase = (1 - self.per_beta) / (time_steps)
            self.per_prior_eps: float = 1e-6
            self.experience_replay: PrioritizedExperienceReplay = (
                make_prioritized_experience_replay(
                    env=env,
                    experience_replay_size=experience_replay_size,
                    batch_size=batch_size,
                    device=device,
                    n_step=n_step,
                    gamma=gamma,
                    network_type=self.policy_net.network_type,
                    alpha=per_alpha,
                )
            )

    def select_action(self, state: Tensor) -> Tensor:
        if self.per_enabled:
            self.adaptive_per_beta()

        self.adaptive_e_greedy()

        sample = random.uniform(0, 1)
        self.steps_done += 1
        if sample > self.epsilon:
            return self.select_greedy_action(state)
        else:
            return self.select_random_action()

    def select_greedy_action(self, state: Tensor, eval: bool = False) -> Tensor:
        if eval:
            self.policy_net.eval()
        with torch.no_grad():
            if self.policy_net.action_type == "discrete":
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                max_valued_action: Tensor = self.policy_net(state)[0].max(1)[1]
                action = max_valued_action.view(1, 1)
            elif self.policy_net.action_type == "multidiscrete":
                action = torch.tensor(
                    self.decode_gym_action(self.policy_net(state)),
                    device=self.device,
                    dtype=torch.long,
                )

            self.policy_net.train()
            return action

    def decode_gym_action(self, nn_action_values: Tensor) -> list:
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        decoded_batch_actions = []
        for dim in range(self.action_dim):
            max_valued_sub_action: Tensor = nn_action_values[dim].max(1)[1]

            decoded_batch_actions.append(max_valued_sub_action.view(1, 1).item())

        return decoded_batch_actions

    def adaptive_e_greedy(self):
        self.writer.epsilon = self.epsilon
        if self.epsilon > self.epsilon_end + 1e-10:
            self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end

    def adaptive_per_beta(self):
        if self.per_beta < 1:
            self.per_beta += self.per_beta_increase
        if self.per_beta > 1:
            self.per_beta = 1

    def optimize_model(self, time_step: int):
        self.policy_net.train()
        if len(self.experience_replay) < self.experience_replay.batch_size:
            return

        for _ in range(self.gradient_steps):
            if self.per_enabled:
                transitions_weights_indices = self.get_per_transitions()
                transitions = transitions_weights_indices[0]
                weights = transitions_weights_indices[1]
                indices = transitions_weights_indices[2]

                elementwise_loss = self.compute_loss(*transitions, elementwise=True)
                total_loss = torch.mean(elementwise_loss * weights)

                self.per_update(elementwise_loss, indices)

            else:
                transitions = self.get_transitions()
                total_loss = self.compute_loss(*transitions)

            self.update_parameters(total_loss, time_step)

    def per_update(self, elementwise_loss: Tensor, indices: list[int]):
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.per_prior_eps
        self.experience_replay.update_priorities(indices, new_priorities)

    def get_transitions(self):
        transitions: Transition = self.experience_replay.sample()

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

    def get_per_transitions(self):
        sample_tuple: tuple[Transition, Tensor] = self.experience_replay.sample(
            self.per_beta
        )
        transitions: Transition = sample_tuple[0]
        weights: Tensor = sample_tuple[1]
        indices: list[int] = sample_tuple[2]

        state_batch = transitions.state.squeeze(1)
        next_state_batch = transitions.next_state.squeeze(1)
        action_batch = transitions.action.squeeze(1)
        reward_batch = transitions.reward.squeeze(1)
        done_batch = transitions.done.squeeze(1).int()
        mask_batch = 1 - done_batch

        return [
            (
                state_batch,
                next_state_batch,
                action_batch,
                reward_batch,
                mask_batch,
            ),
            weights,
            indices,
        ]

    def compute_loss(
        self,
        state_batch: Tensor,
        next_state_batch: Tensor,
        action_batch: Tensor,
        reward_batch: Tensor,
        mask_batch: Tensor,
        elementwise: bool = False,
    ) -> Tensor:
        if elementwise:
            total_loss: Tensor = torch.zeros_like(action_batch, dtype=torch.float32)
            criterion = nn.SmoothL1Loss(reduction="none")
        else:
            total_loss: Tensor = 0.0
            criterion = nn.SmoothL1Loss()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        for sub_action in range(self.action_dim):
            state_action_values = self.policy_net(state_batch)[sub_action].gather(
                1, action_batch.transpose(0, 1)[sub_action].view(-1, 1)
            )

            if self.double_enabled:
                next_state_values = self.ddqn_values(
                    sub_action, state_batch, next_state_batch, mask_batch
                )
            else:
                next_state_values = self.dqn_values(
                    sub_action, next_state_batch, mask_batch
                )

            target_state_action_values: Tensor = (
                next_state_values * self.gamma**self.experience_replay.n_step
            ) + reward_batch.squeeze(1)

            total_loss += criterion(
                state_action_values, target_state_action_values.unsqueeze(1)
            )

        if self.per_enabled:
            self.writer.losses.append(torch.mean(total_loss).item())
        else:
            self.writer.losses.append(total_loss.item())

        return total_loss

    def dqn_values(
        self,
        sub_action: int,
        next_state_batch: Tensor,
        mask_batch: Tensor,
    ):
        next_state_values = torch.zeros(
            self.experience_replay.batch_size, device=self.device
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch)[sub_action].max(1)[
                0
            ] * mask_batch.squeeze(1)

        return next_state_values

    def ddqn_values(
        self,
        sub_action: int,
        state_batch: Tensor,
        next_state_batch: Tensor,
        mask_batch: Tensor,
    ):
        selected_action = self.policy_net(state_batch)[sub_action].argmax(
            dim=1, keepdim=True
        )

        next_state_values = torch.zeros(
            self.experience_replay.batch_size, device=self.device
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch)[sub_action].gather(
                1, selected_action
            ).squeeze(1) * mask_batch.squeeze(1)

        return next_state_values

    def update_parameters(self, total_loss: Tensor, time_step: int):
        self.optimizer.zero_grad()
        total_loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=self.gradient_clipping_max_norm
        )
        self.optimizer.step()

        if time_step % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.noisy_enabled:
                # Noisy reset noise
                self.policy_net.reset_noise()
                self.target_net.reset_noise()
