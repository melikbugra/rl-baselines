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
from rl_baselines.policy_based.ppo.ppo_writer import PPOWriter
from rl_baselines.utils.replay_buffers import TransitionBuffer, make_transition_buffer


class PPOAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        gamma: float,
        time_steps: int,
        experience_replay_type: str,
        # base agent attributes
        neural_network: BaseNeuralNetwork,
        writer: PPOWriter,
        learning_rate: float = None,
        device: str = None,
        gradient_clipping_max_norm: float = 1.0,
        # optional ppo attributes
        n_epochs: int = 10,
        clip_range: float = 0.2,
        batch_size: int = 5,
        gae_lambda: float = 0.95,
    ) -> None:
        super().__init__(
            env=env,
            neural_network=neural_network,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
            writer=writer,
            learning_rate=learning_rate,
            device=device,
        )

        self.writer: PPOWriter = writer

        self.net: BaseNeuralNetwork = neural_network

        self.gamma = gamma

        self.time_steps: int = time_steps

        if experience_replay_type == "tb":
            self.experience_replay: TransitionBuffer = make_transition_buffer(
                device=device,
            )

        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.memory_max_size = batch_size * n_epochs

        self.log_probs: list[Tensor] = []
        self.state_values: list[Tensor] = []

    def select_action(self, state: Tensor) -> Tensor:
        state = state.float()
        outs, state_value = self.net(state)

        if self.net.action_type == "discrete":
            action_probs = torch.softmax(outs[0], dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            if self.net.training:
                self.log_probs.append(log_prob.detach())
                self.state_values.append(state_value[0].detach())

            return action

        elif self.net.action_type == "multidiscrete":
            action_probs = [nn.Softmax(dim=-1)(action_value) for action_value in outs]
            actions = []
            action_log_probs = []
            for probs in action_probs:
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                actions.append(action)

                log_prob = action_dist.log_prob(action)
                action_log_probs.append(log_prob)

            if self.net.training:
                self.log_probs.append(sum(action_log_probs.detach()))
                self.state_values.append(state_value[0].detach())

            return torch.tensor(actions, device=self.device, dtype=torch.long)

        elif self.net.action_type == "continuous":
            actions = []
            log_probs = []
            action_range = torch.tensor(
                (self.env.action_space.high - self.env.action_space.low) / 2.0,
                dtype=torch.float32,
                device=self.device,
            )
            action_mid = torch.tensor(
                (self.env.action_space.high + self.env.action_space.low) / 2.0,
                dtype=torch.float32,
                device=self.device,
            )
            for mean, std in outs:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                tanh_action = torch.tanh(action)
                scaled_action = action_mid + action_range * tanh_action
                actions.append(action)
                log_probs.append(log_prob)

            joint_log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)

            if self.net.training:
                self.log_probs.append(joint_log_prob.detach())
                self.state_values.append(state_value[0].detach())

            return torch.stack(actions).to(self.device).float().squeeze(1)

    def select_greedy_action(self, state: Tensor, eval: bool = False) -> Tensor:
        if eval:
            self.net.eval()
        with torch.no_grad():
            action = self.select_action(state)
        if eval:
            self.net.train()
        return action

    def optimize_model(self, time_step: int):
        if len(self.experience_replay) < self.memory_max_size:
            return

        for _ in range(self.n_epochs):
            transitions = self.get_transitions()

            for total_loss in self.compute_loss(*transitions):
                self.update_parameters(total_loss)

        self.log_probs = []
        self.state_values = []

        self.experience_replay.clear()  # clear the replay buffer after sampling because it is an on-policy algorithm

    def get_transitions(self):
        transitions, mini_batches = self.experience_replay.mini_batched_sample(
            self.batch_size
        )

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
            mini_batches,
        )

    def compute_loss(
        self,
        state_batch: Tensor,
        next_state_batch: Tensor,
        action_batch: Tensor,
        reward_batch: Tensor,
        mask_batch: Tensor,
        mini_batches: list,
    ):
        returns, advantages = self.compute_returns_advantages(
            reward_batch, mask_batch, next_state_batch
        )

        log_probs = torch.cat(self.log_probs).unsqueeze(1)

        for mini_batch in mini_batches:
            states = state_batch[mini_batch]
            old_log_probs = log_probs[mini_batch]
            actions = action_batch[mini_batch]

            outs, state_values = self.net(states.float())

            state_values = state_values[0]

            if self.net.action_type == "discrete":
                action_probs = torch.softmax(outs[0], dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)

                new_log_probs = action_dist.log_prob(actions.squeeze(1))
            elif self.net.action_type == "multidiscrete":
                action_probs = [
                    nn.Softmax(dim=-1)(action_value) for action_value in outs
                ]
                new_log_probs = []
                for probs, action in zip(action_probs, actions):
                    action_dist = torch.distributions.Categorical(probs)
                    new_log_probs.append(action_dist.log_prob(action))
                new_log_probs = sum(new_log_probs)
            elif self.net.action_type == "continuous":
                new_log_probs = []
                for mean, std in outs:
                    dist = torch.distributions.Normal(mean, std)
                    new_log_probs.append(dist.log_prob(actions.squeeze(1)))
                new_log_probs = torch.stack(new_log_probs, dim=-1).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs.squeeze(1))
            weighted_log_probs = advantages[mini_batch] * ratio
            weighted_clipped_log_probs = (
                torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                * advantages[mini_batch]
            )

            entropy = dist.entropy().sum(axis=-1).mean()

            actor_loss = -torch.min(
                weighted_log_probs, weighted_clipped_log_probs
            ).mean()

            mini_batch_returns = returns[mini_batch].squeeze(1)

            critic_loss = (mini_batch_returns - state_values.squeeze(1)) ** 2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.writer.actor_losses.append(actor_loss.item())
            self.writer.critic_losses.append(critic_loss.item())

            yield total_loss

    def compute_returns_advantages(
        self, reward_batch: Tensor, mask_batch: Tensor, next_state_batch: Tensor
    ) -> tuple[Tensor, Tensor]:
        T = len(reward_batch)
        advantages = np.zeros(T, dtype=np.float32)

        # Append bootstrap value if needed (for T+1 state values)
        if len(self.state_values) == T:
            if mask_batch[-1].item() == 1:
                last_next_state = next_state_batch[-1].float().unsqueeze(0)
                _, bootstrap_value = self.net(last_next_state)
                bootstrap_value = bootstrap_value[0].detach()
            else:
                bootstrap_value = torch.zeros(1, device=self.device).unsqueeze(0)
            self.state_values.append(bootstrap_value)

        for t in range(T):
            discount = 1
            a_t = 0
            for k in range(t, T):
                delta = (
                    reward_batch[k].item()
                    + self.gamma
                    * self.state_values[k + 1].item()
                    * mask_batch[k].item()
                    - self.state_values[k].item()
                )
                a_t += discount * delta
                discount *= self.gamma * self.gae_lambda
            advantages[t] = a_t

        advantages = torch.tensor(advantages, device=self.device)
        state_values = torch.cat(self.state_values)  # expects T+1 values
        returns = advantages.unsqueeze(1) + state_values[:-1]

        return returns, advantages

    def update_parameters(self, total_loss: Tensor):
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), max_norm=self.gradient_clipping_max_norm
        )
        self.optimizer.step()

    def decode_gym_action(self, nn_outs):
        return super().decode_gym_action(nn_outs)
