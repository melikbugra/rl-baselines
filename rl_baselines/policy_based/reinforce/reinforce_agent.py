from copy import deepcopy

from gymnasium import Env
import numpy as np
import random
from torch import Tensor
import torch

import torch.nn as nn

from rl_baselines.utils.base_classes import (
    BaseAgent,
    BaseExperienceReplay,
    BaseNeuralNetwork,
    Transition,
)
from rl_baselines.policy_based.reinforce.reinforce_writer import ReinforceWriter
from rl_baselines.utils.replay_buffers import TransitionBuffer, make_transition_buffer


class ReinforceAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        gamma: float,
        episodes_to_train: int,
        experience_replay_type: str,
        # base agent attributes
        neural_network: BaseNeuralNetwork,
        writer: ReinforceWriter,
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
        self.writer: ReinforceWriter = writer

        self.net: BaseNeuralNetwork = neural_network

        self.gamma = gamma

        self.episodes_to_train: int = episodes_to_train

        if experience_replay_type == "tb":
            self.experience_replay: TransitionBuffer = make_transition_buffer(
                device=device,
            )

        self.gradient_clipping_max_norm: float = gradient_clipping_max_norm

        self.log_probs: list[Tensor] = []

    def select_action(self, state: Tensor) -> Tensor:
        """Selects an action under exploration strategy

        :param state: Environment state as a tensor
        :type state: Tensor
        :return: Chosen action as a tensor
        :rtype: Tensor
        """
        state = state.float()
        outs = self.net(state)

        if self.net.action_type == "discrete":
            action_probs = torch.softmax(outs[0], dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            if self.net.training:
                self.log_probs.append(log_prob)

            return action

        elif self.net.action_type == "multidiscrete":
            action_probs = [nn.Softmax(dim=-1)(action_value) for action_value in outs]
            actions = []
            log_probs = []
            for probs in action_probs:
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                actions.append(action.item())

                log_prob = action_dist.log_prob(action)
                log_probs.append(log_prob)
            if self.net.training:
                self.log_probs.append(torch.cat(log_probs))
            return torch.tensor(actions, device=self.device, dtype=torch.long)
        elif self.net.action_type == "continuous":
            actions = []
            log_probs = []
            for mean, std in outs:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                actions.append(action.item())
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)
            if self.net.training:  # Only store log_probs during training
                self.log_probs.append(torch.cat(log_probs))

            return torch.tensor(actions, device=self.device, dtype=torch.float32)

    def select_greedy_action(self, state: Tensor, eval: bool = False) -> Tensor:
        if eval:
            self.net.eval()
        with torch.no_grad():
            action = self.select_action(state)
        if eval:
            self.net.train()
        return action

    def optimize_model(self, time_step):
        self.net.train()

        transitions = self.get_transitions()

        total_loss = self.compute_loss(*transitions)

        self.update_parameters(total_loss, time_step)

        self.log_probs = []

    def get_transitions(self):
        transitions: Transition = (
            self.experience_replay.sample()
        )  # it will always return a single episode transitions because batch_size is 1
        self.experience_replay.clear()  # clear the replay buffer after sampling because it is an on-policy algorithm

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
    ):
        returns = self.compute_returns(reward_batch)
        log_probs = self.log_probs
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        self.writer.losses.append(loss.item())

        return loss

    def compute_returns(self, reward_batch: Tensor) -> Tensor:
        rewards = reward_batch.squeeze(1).tolist()
        returns = []
        G = 0
        # Ödülleri tersten dolaş:
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def update_parameters(self, total_loss: Tensor, time_step: int):
        self.optimizer.zero_grad()
        total_loss.backward()
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_norm_(
        #     self.net.parameters(), max_norm=self.gradient_clipping_max_norm
        # )
        self.optimizer.step()

    def decode_gym_action(self, nn_outs):
        return super().decode_gym_action(nn_outs)
