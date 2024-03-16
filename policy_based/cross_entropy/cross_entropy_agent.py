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
from policy_based.cross_entropy.cross_entropy_writer import CrossEntropyWriter
from utils.buffer import ExperienceReplay, make_experience_replay


class CrossEntropyAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        percentile: int,
        episodes_to_train: int,
        experience_replay_type: str,
        experience_replay_size: int,
        batch_size: int,
        # base agent attributes
        neural_network: BaseNeuralNetwork,
        writer: CrossEntropyWriter,
        learning_rate: float = None,
        device: str = None,
        gradient_clipping_max_norm: float = 1.0,
    ) -> None:
        super().__init__(
            env=env,
            neural_network=neural_network,
            writer=writer,
            learning_rate=learning_rate,
            device=device,
        )
        self.writer: CrossEntropyWriter = writer

        self.net: BaseNeuralNetwork = neural_network

        self.percentile: int = percentile
        self.episodes_to_train: int = episodes_to_train

        self.criterion = nn.CrossEntropyLoss()
        self.sm = nn.Softmax(dim=1)

        if experience_replay_type == "er":
            self.experience_replay: ExperienceReplay = make_experience_replay(
                env=env,
                experience_replay_size=experience_replay_size,
                batch_size=batch_size,
                device=device,
                network_type=self.net.network_type,
            )

        self.gradient_clipping_max_norm: float = gradient_clipping_max_norm

    def select_action(self, state: Tensor) -> Tensor:
        self.net.eval()
        with torch.no_grad():
            if self.net.action_type == "discrete":
                act_probs_v = self.sm(self.net(state)[0])
                act_probs = act_probs_v.data.cpu().numpy()[0]
                return torch.tensor(
                    np.random.choice(len(act_probs), p=act_probs),
                    device=self.device,
                    dtype=torch.long,
                )
            elif self.net.action_type == "multidiscrete":
                return torch.tensor(
                    self.decode_gym_action(self.net(state)),
                    device=self.device,
                    dtype=torch.long,
                )

    def select_greedy_action(self, state: Tensor) -> Tensor:
        return self.select_action(state=state)

    def decode_gym_action(self, nn_action_values: Tensor) -> list:
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        decoded_batch_actions = []
        for dim in range(self.action_dim):
            max_valued_sub_action: Tensor = nn_action_values[dim].max(1)[1]

            decoded_batch_actions.append(max_valued_sub_action.view(1, 1).item())

        return decoded_batch_actions

    def optimize_model(self, time_step: int):
        self.net.train()
        if len(self.experience_replay) < self.experience_replay.batch_size:
            return
        if (
            len(self.experience_replay.done_buffer.squeeze().nonzero(as_tuple=True)[0])
            < self.episodes_to_train
        ):
            return

        transitions = self.get_transitions()

        total_loss = self.compute_loss(*transitions)

        self.update_parameters(total_loss, time_step)

    def get_transitions(self):
        # Calculate episode scores
        episode_boundaries = self.experience_replay.done_buffer.squeeze().nonzero(
            as_tuple=True
        )[0]
        episode_scores = []
        start_idx = 0
        for end_idx in episode_boundaries:
            episode_score = self.experience_replay.reward_buffer[
                start_idx : end_idx + 1
            ].sum()
            episode_scores.append((start_idx, end_idx, episode_score.item()))
            start_idx = end_idx + 1

        # Sort episodes by their score
        sorted_episodes = sorted(episode_scores, key=lambda x: x[2])

        # Calculate the 70th percentile index
        percentile_idx = int(len(sorted_episodes) * 0.7)

        # Select episodes above the 70th percentile
        selected_episodes = sorted_episodes[percentile_idx:]

        if selected_episodes:
            self.writer.score_bound = selected_episodes[0][2]

        # Flatten selected indices (assuming episodes can be of different lengths, this might need adjustment)
        selected_indices = []
        for start_idx, end_idx, _ in selected_episodes:
            selected_indices.extend(range(start_idx, end_idx + 1))

        # # Sample from selected indices
        # if len(selected_indices) < self.experience_replay.batch_size:
        #     # Not enough samples, sample with replacement or adjust batch size
        #     idxs = np.random.choice(
        #         selected_indices, size=self.experience_replay.batch_size, replace=True
        #     )
        # else:
        #     idxs = np.random.choice(
        #         selected_indices, size=self.experience_replay.batch_size, replace=False
        #     )

        state = self.experience_replay.state_buffer[selected_indices]
        action = self.experience_replay.action_buffer[selected_indices]
        next_state = torch.stack(
            [self.experience_replay.next_state_buffer[idx] for idx in selected_indices]
        )
        reward = self.experience_replay.reward_buffer[selected_indices]
        done = self.experience_replay.done_buffer[selected_indices]
        transitions = Transition(
            state=state, action=action, next_state=next_state, reward=reward, done=done
        )

        state_batch = transitions.state.squeeze(1)
        next_state_batch = transitions.next_state.squeeze(1)
        action_batch = transitions.action.squeeze(1)
        reward_batch = transitions.reward.squeeze(1)
        done_batch = transitions.done.squeeze(1).int()
        mask_batch = 1 - done_batch

        self.experience_replay.clear()

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

        for sub_action in range(self.action_dim):
            action_probs = self.net(state_batch)[sub_action]
            total_loss = self.criterion(action_probs, action_batch.squeeze())

        self.writer.losses.append(total_loss.item())

        return total_loss

    def update_parameters(self, total_loss: Tensor, time_step: int):
        self.optimizer.zero_grad()
        total_loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), max_norm=self.gradient_clipping_max_norm
        )
        self.optimizer.step()
