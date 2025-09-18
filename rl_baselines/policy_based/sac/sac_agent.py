import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Normal
from gymnasium import Env

from rl_baselines.utils.base_classes import (
    BaseAgent,
    BaseNeuralNetwork,
    BaseSACNeuralNetwork,
    Transition,
)
from rl_baselines.utils.replay_buffers import make_experience_replay
from rl_baselines.policy_based.sac.sac_writer import SACWriter


class SACAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        writer: SACWriter,
        experience_replay_type: str,
        experience_replay_size: int,
        batch_size: int,
        learning_rate: float,
        device: str,
        gradient_clipping_max_norm: float,
        neural_network,
        tau: float,
        gamma: float,
        target_entropy: float,
        learning_starts: int,
        gradient_steps: int,
    ):
        super().__init__(
            env=env,
            neural_network=neural_network,
            writer=writer,
            learning_rate=learning_rate,
            device=device,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
        )

        self.net: BaseSACNeuralNetwork = neural_network

        self.writer: SACWriter = writer
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_entropy = target_entropy
        self.learning_starts = learning_starts
        self.gradient_steps = gradient_steps

        self.log_alpha = torch.tensor(
            0.0, dtype=torch.float32, requires_grad=True, device=device
        )

        self.actor_optimizer = torch.optim.Adam(
            self.net.actor.parameters(), lr=learning_rate
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.net.critic1.parameters(), lr=learning_rate
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.net.critic2.parameters(), lr=learning_rate
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)

        self.experience_replay = make_experience_replay(
            env=env,
            experience_replay_size=experience_replay_size,
            batch_size=batch_size,
            device=self.device,
            network_type=self.net.network_type,
            action_type=self.net.action_type,
        )

        self.act_low = float(env.action_space.low[0])
        self.act_high = float(env.action_space.high[0])
        self.max_action = max(abs(self.act_low), abs(self.act_high))

    def select_action(self, state):
        if self.steps_done < self.learning_starts:
            self.steps_done += 1
            return self.select_random_action()

        state = state.float()
        outs, _, _, _, _ = self.net(state=state, actor_pass=True)

        if self.net.action_type == "continuous":
            mean, std = outs[0]

            if self.net.training:
                noise = torch.randn_like(mean)
                z = mean + std * noise
            else:
                z = mean

            action = torch.tanh(z) * self.max_action

        return action.detach()

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

    def optimize_model(self, time_step):
        if self.steps_done < self.learning_starts:
            return
        if len(self.experience_replay) < self.batch_size:
            return

        grad_updates = max(1, int(self.gradient_steps))

        for _ in range(grad_updates):
            transitions = self.get_transitions()

            actor_loss, critic1_loss, critic2_loss, alpha_loss = self.compute_losses(
                *transitions
            )

            self.update_parameters(actor_loss, critic1_loss, critic2_loss, alpha_loss)

    def compute_losses(
        self, state_batch, action_batch, next_state_batch, reward_batch, mask_batch
    ):
        with torch.no_grad():
            outs, _, _, _, _ = self.net(state=next_state_batch, actor_pass=True)
            next_mean, next_std = outs[0]
            noise = torch.randn_like(next_mean)
            next_z = next_mean + next_std * noise
            next_action = torch.tanh(next_z) * self.max_action

            log_prob_gauss = -0.5 * (
                ((next_z - next_mean) / next_std) ** 2
                + 2 * torch.log(next_std)
                + np.log(2 * np.pi)
            )
            log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
            log_prob_policy = log_prob_gauss - (
                1 - torch.tanh(next_z) ** 2 + 1e-6
            ).log().sum(dim=-1, keepdim=True)

            _, _, _, target_q1, target_q2 = self.net(
                state=next_state_batch,
                action=next_action,
                target_pass=True,
            )
            target_min_q = torch.min(target_q1, target_q2)
            y = reward_batch + mask_batch * self.gamma * (
                target_min_q - torch.exp(self.log_alpha) * log_prob_policy
            )

        # Critic loss
        _, current_q1, current_q2, _, _ = self.net(
            state=state_batch,
            action=action_batch,
            critic_pass=True,
        )

        critic1_loss = F.mse_loss(current_q1, y)
        critic2_loss = F.mse_loss(current_q2, y)
        self.writer.critic_losses.append((critic1_loss + critic2_loss).item())

        # Actor loss
        outs, _, _, _, _ = self.net(state=state_batch, actor_pass=True)
        mean, std = outs[0]
        noise = torch.randn_like(mean)
        z = mean + std * noise
        action_sample = torch.tanh(z) * self.max_action
        log_prob_gauss = -0.5 * (
            ((z - mean) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi)
        )
        log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
        log_prob_policy = log_prob_gauss - (1 - torch.tanh(z) ** 2 + 1e-6).log().sum(
            dim=-1, keepdim=True
        )

        _, q1_pi, q2_pi, _, _ = self.net(
            state=state_batch,
            action=action_sample,
            critic_pass=True,
        )
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (torch.exp(self.log_alpha) * log_prob_policy - min_q_pi).mean()
        self.writer.actor_losses.append(actor_loss.item())

        # Alpha loss
        alpha_loss = -(
            self.log_alpha * (log_prob_policy + self.target_entropy).detach()
        ).mean()
        self.writer.alpha_losses.append(alpha_loss.item())

        return actor_loss, critic1_loss, critic2_loss, alpha_loss

    def update_parameters(
        self,
        actor_loss: Tensor,
        critic1_loss: Tensor,
        critic2_loss: Tensor,
        alpha_loss: Tensor,
    ):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.gradient_clipping_max_norm:
            torch.nn.utils.clip_grad_norm_(
                self.net.actor.parameters(), self.gradient_clipping_max_norm
            )
        self.actor_optimizer.step()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        if self.gradient_clipping_max_norm:
            torch.nn.utils.clip_grad_norm_(
                self.net.critic1.parameters(), self.gradient_clipping_max_norm
            )
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.gradient_clipping_max_norm:
            torch.nn.utils.clip_grad_norm_(
                self.net.critic2.parameters(), self.gradient_clipping_max_norm
            )
        self.critic2_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # soft update target networks
        for param, target_param in zip(
            self.net.critic1.parameters(), self.net.target_critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.net.critic2.parameters(), self.net.target_critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

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
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
        )

    def decode_gym_action(self, nn_action_values):
        return super().decode_gym_action(nn_action_values)
