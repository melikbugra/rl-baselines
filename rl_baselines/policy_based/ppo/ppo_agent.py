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
        gradient_clipping_max_norm: float = None,
        # optional ppo attributes
        n_epochs: int = 10,
        clip_range: float = 0.2,
        batch_size: int = 5,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        memory_size: int = 2048,
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

        if self.net.action_type == "continuous":
            self.action_low = torch.tensor(
                self.env.action_space.low, dtype=torch.float32, device=self.device
            )
            self.action_high = torch.tensor(
                self.env.action_space.high, dtype=torch.float32, device=self.device
            )

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
        self.memory_max_size = memory_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.log_probs: list[Tensor] = []
        self.state_values: list[Tensor] = []

    def select_action(self, state: Tensor) -> Tensor:
        state = state.float()
        outs, state_value = self.net(state)

        if self.net.action_type == "discrete":
            action_probs = outs[0]

            if self.net.training:
                action_dist = torch.distributions.Categorical(logits=action_probs)
                action = action_dist.sample()

                log_prob = action_dist.log_prob(action)

                self.log_probs.append(log_prob.detach().cpu().item())
                self.state_values.append(state_value[0].item())
            else:
                action = torch.argmax(action_probs, dim=-1)

            return action

        elif self.net.action_type == "multidiscrete":
            # action_probs = [nn.Softmax(dim=-1)(action_value) for action_value in outs]
            # actions = []
            # action_log_probs = []
            # for probs in action_probs:
            #     action_dist = torch.distributions.Categorical(probs)
            #     action = action_dist.sample()
            #     actions.append(action)

            #     log_prob = action_dist.log_prob(action)
            #     action_log_probs.append(log_prob)

            # if self.net.training:
            #     self.log_probs.append(sum(action_log_probs.detach()))
            #     self.state_values.append(state_value[0].detach())

            # return torch.tensor(actions, device=self.device, dtype=torch.long)
            raise NotImplementedError

        elif self.net.action_type == "continuous":

            if self.net.training:
                mean, std = outs[0]
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()
                log_prob = dist.log_prob(action).sum(axis=-1)

                self.log_probs.append(log_prob.detach().cpu().item())
                self.state_values.append(state_value[0].item())
            else:
                with torch.no_grad():
                    outs, _ = self.net(state)
                    mean, _ = outs[0]
                action = mean
            return torch.clip(action, self.action_low, self.action_high).detach()

    def select_greedy_action(self, state: Tensor, eval: bool = False) -> Tensor:
        if eval:
            self.net.eval()
            self.net.training = False
        with torch.no_grad():
            action = self.select_action(state)
        if eval:
            self.net.train()
            self.net.training = True
        return action

    def optimize_model(self, time_step: int):
        if len(self.experience_replay) < self.memory_max_size:
            return

        transitions = self.get_transitions()

        for _ in range(self.n_epochs):
            for total_loss in self.compute_loss(*transitions):
                self.update_parameters(total_loss)

        self.log_probs = []
        self.state_values = []

        self.experience_replay.clear()  # clear the replay buffer after sampling because it is an on-policy algorithm

    def get_transitions(self):
        transitions, mini_batches = self.experience_replay.mini_batched_sample(
            self.batch_size
        )

        state_batch = transitions.state
        next_state_batch = transitions.next_state
        action_batch = transitions.action
        reward_batch = transitions.reward.squeeze()
        done_batch = transitions.done.squeeze().int()
        mask_batch = 1 - done_batch

        advantages, returns = self.compute_returns_advantages(
            reward_batch, mask_batch, next_state_batch
        )

        log_probs = torch.tensor(
            self.log_probs, dtype=torch.float32, device=self.device
        )

        return (
            state_batch,
            next_state_batch,
            action_batch,
            reward_batch,
            mask_batch,
            advantages,
            returns,
            log_probs,
            mini_batches,
        )

    def compute_loss(
        self,
        state_batch: Tensor,
        next_state_batch: Tensor,
        action_batch: Tensor,
        reward_batch: Tensor,
        mask_batch: Tensor,
        advantages_tensor: Tensor,
        returns_tensor: Tensor,
        log_probs_tensor: Tensor,
        mini_batches: list,
    ):
        for mini_batch in mini_batches:
            states = state_batch[mini_batch]
            old_log_probs = log_probs_tensor[mini_batch]
            actions = action_batch[mini_batch]

            outs, state_values = self.net(states.float())

            state_values = state_values[0]

            if self.net.action_type == "discrete":
                action_probs = outs[0].squeeze(1)
                action_dist = torch.distributions.Categorical(logits=action_probs)

                new_log_probs = action_dist.log_prob(actions.squeeze(1))
                entropy = action_dist.entropy()
            elif self.net.action_type == "multidiscrete":
                action_probs = [
                    nn.Softmax(dim=-1)(action_value) for action_value in outs
                ]
                new_log_probs = []
                for probs, action in zip(action_probs, actions):
                    action_dist = torch.distributions.Categorical(probs)
                    new_log_probs.append(action_dist.log_prob(action))
                new_log_probs = sum(new_log_probs)
                entropy = dist.entropy()
            elif self.net.action_type == "continuous":
                mean, std = outs[0]
                dist = torch.distributions.Normal(mean.squeeze(1), std)
                new_log_probs = dist.log_prob(actions.flatten(1)).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1)

            ratio = torch.exp(new_log_probs - old_log_probs.squeeze())

            mini_batch_advantages = advantages_tensor[mini_batch]

            weighted_log_probs = mini_batch_advantages * ratio
            weighted_clipped_log_probs = (
                torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                * mini_batch_advantages
            )

            actor_loss = -torch.min(
                weighted_log_probs, weighted_clipped_log_probs
            ).mean()

            mini_batch_returns = returns_tensor[mini_batch]

            critic_loss = ((mini_batch_returns - state_values) ** 2).mean()

            total_loss = (
                actor_loss
                + self.value_coef * critic_loss
                - self.entropy_coef * entropy.mean()
            )

            self.writer.actor_losses.append(actor_loss.item())
            self.writer.critic_losses.append(critic_loss.item())

            yield total_loss

    def compute_returns_advantages(
        self, reward_batch: Tensor, mask_batch: Tensor, next_state_batch: Tensor
    ) -> tuple[Tensor, Tensor]:
        if len(self.state_values) == len(reward_batch):
            # Append last_value to bootstrap the value estimates
            if mask_batch[-1].item() == 1:
                last_next_state = next_state_batch[-1].float()
                _, bootstrap_value = self.net(last_next_state)
                bootstrap_value = bootstrap_value[0].detach()
            else:
                bootstrap_value = torch.zeros(1, device=self.device)
            self.state_values = self.state_values + [bootstrap_value.item()]
        else:
            pass

        advantages = []
        gae = 0

        for t in reversed(range(len(reward_batch))):
            mask = mask_batch[t].item()
            delta = (
                reward_batch[t].item()
                + self.gamma * self.state_values[t + 1] * mask
                - self.state_values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, self.state_values[:-1])]

        returns_tensor = torch.as_tensor(
            np.array(returns), dtype=torch.float32, device=self.device
        )
        advantages_tensor = torch.as_tensor(
            np.array(advantages), dtype=torch.float32, device=self.device
        )
        # Normalize advantages.
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        return advantages_tensor, returns_tensor

    def update_parameters(self, total_loss: Tensor):
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.gradient_clipping_max_norm:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), max_norm=self.gradient_clipping_max_norm
            )
        self.optimizer.step()

    def decode_gym_action(self, nn_outs):
        return super().decode_gym_action(nn_outs)
