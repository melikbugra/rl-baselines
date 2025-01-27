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
from rl_baselines.policy_based.a2c.a2c_writer import A2CWriter
from rl_baselines.utils.replay_buffers import TransitionBuffer, make_transition_buffer


class A2CAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        gamma: float,
        time_steps: int,
        experience_replay_type: str,
        # base agent attributes
        neural_network: BaseNeuralNetwork,
        writer: A2CWriter,
        learning_rate: float = None,
        device: str = None,
        gradient_clipping_max_norm: float = 1.0,
        # optional a2c attributes
        n_step: int = 5,
    ) -> None:
        super().__init__(
            env=env,
            neural_network=neural_network,
            writer=writer,
            learning_rate=learning_rate,
            device=device,
        )

        self.writer: A2CWriter = writer

        self.net: BaseNeuralNetwork = neural_network

        self.gamma = gamma

        self.time_steps: int = time_steps

        if experience_replay_type == "tb":
            self.experience_replay: TransitionBuffer = make_transition_buffer(
                device=device,
            )

        self.gradient_clipping_max_norm: float = gradient_clipping_max_norm

        self.n_step: int = n_step

        self.log_probs: list[Tensor] = []
        self.state_values: list[Tensor] = []

    def select_action(self, state: Tensor) -> Tensor:
        """Selects an action under exploration strategy

        :param state: Environment state as a tensor
        :type state: Tensor
        :return: Chosen action as a tensor
        :rtype: Tensor
        """
        state = state.float()
        outs, state_value = self.net(state)

        if self.net.action_type == "discrete":
            action_probs = torch.softmax(outs[0], dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            if self.net.training:
                self.log_probs.append(log_prob)
                self.state_values.append(state_value)

            return action

        elif self.net.action_type == "multidiscrete":
            action_probs = [nn.Softmax(dim=-1)(action_value) for action_value in outs]
            actions = []
            action_log_probs = []
            for probs in action_probs:
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                actions.append(action.item())

                log_prob = action_dist.log_prob(action)
                action_log_probs.append(log_prob)
            if self.net.training:
                self.log_probs.append(sum(action_log_probs))
                self.state_values.append(state_value)
            return torch.tensor(actions, device=self.device, dtype=torch.long)

        elif self.net.action_type == "continuous":
            actions = []
            log_probs = []
            action_range = torch.tensor(
                (self.env.action_space.high - self.env.action_space.low) / 2.0,
                dtype=torch.float32,
            )
            action_mid = torch.tensor(
                (self.env.action_space.high + self.env.action_space.low) / 2.0,
                dtype=torch.float32,
            )
            for mean, std in outs:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                tanh_action = torch.tanh(action)
                scaled_action = action_mid + action_range * tanh_action
                actions.append(scaled_action)
                log_probs.append(log_prob)
            if self.net.training:  # Only store log_probs during training
                self.log_probs.append(sum(log_probs))
                self.state_values.append(state_value)

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
        if len(self.experience_replay) == self.n_step:

            self.net.train()

            transitions = self.get_transitions()

            total_loss = self.compute_loss(*transitions)

            self.update_parameters(total_loss, time_step)

            self.log_probs = []
            self.state_values = []

            self.experience_replay.clear()  # clear the replay buffer after sampling because it is an on-policy algorithm

    def get_transitions(self):
        transitions: Transition = (
            self.experience_replay.sample()
        )  # it will return a Transition object including n-step transitions

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
        returns, advantages = self.compute_returns_advantages(
            reward_batch, mask_batch, next_state_batch
        )

        log_probs = torch.cat(self.log_probs).unsqueeze(1)
        state_values = torch.cat(self.state_values)

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(state_values, returns)

        total_loss = actor_loss + critic_loss

        self.writer.actor_losses.append(actor_loss.item())
        self.writer.critic_losses.append(critic_loss.item())

        return total_loss

    def compute_returns_advantages(
        self, reward_batch: Tensor, mask_batch: Tensor, next_state_batch: Tensor
    ) -> tuple[Tensor, Tensor]:
        returns = []
        advantages = []
        next_value = self.net(next_state_batch[-1].unsqueeze(0))[1].detach()
        G = next_value

        for reward, mask, value in zip(
            reversed(reward_batch.squeeze().tolist()),
            reversed(mask_batch.tolist()),
            reversed(self.state_values),
        ):
            G = reward + self.gamma * G * mask
            returns.insert(0, G)
            advantage = G - value.detach()
            advantages.insert(0, advantage)
            next_value = value

        returns = torch.cat(returns).to(self.device)
        advantages = torch.cat(advantages).to(self.device)

        return returns, advantages

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
