import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from gymnasium import Env
from typing import Optional, Callable

from rl_baselines.utils.base_classes import (
    BaseAgent,
    BaseSACNeuralNetwork,
    Transition,
)
from rl_baselines.utils.replay_buffers import (
    make_experience_replay,
    make_hindsight_experience_replay,
    HindsightExperienceReplay,
    GoalConditionedTransition,
)
from rl_baselines.policy_based.sac.sac_writer import SACWriter

MIN_LOG_STD = -20.0
MAX_LOG_STD = 2.0
LOG_TWO = float(np.log(2.0))


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
        neural_network: BaseSACNeuralNetwork,
        tau: float,
        gamma: float,
        target_entropy: float,
        learning_starts: int,
        gradient_steps: int,
        # HER specific parameters
        n_sampled_goal: int = 4,
        goal_selection_strategy: str = "future",
        her_compute_reward: Optional[Callable] = None,
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

        # Temperature (alpha)
        self.log_alpha = torch.tensor(
            0.0, dtype=torch.float32, requires_grad=True, device=device
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)

        # Actor & Critics optimizers (ensemble: tüm başlar tek optimizer)
        self.actor_optimizer = torch.optim.Adam(
            self.net.actor.parameters(), lr=learning_rate
        )
        critic_params = []
        for c in self.net.critics:
            critic_params += list(c.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=learning_rate)

        # Experience Replay Type
        self.experience_replay_type = experience_replay_type
        self.use_her = experience_replay_type == "her"

        # Replay Buffer
        if self.use_her:
            self.experience_replay = make_hindsight_experience_replay(
                env=env,
                experience_replay_size=experience_replay_size,
                batch_size=batch_size,
                device=self.device,
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy=goal_selection_strategy,
                gamma=gamma,
                network_type=self.net.network_type,
                action_type=self.net.action_type,
                compute_reward=her_compute_reward,
            )
        else:
            self.experience_replay = make_experience_replay(
                env=env,
                experience_replay_size=experience_replay_size,
                batch_size=batch_size,
                device=self.device,
                network_type=self.net.network_type,
                action_type=self.net.action_type,
            )

        # Action scaling info
        self.act_low = float(env.action_space.low[0])
        self.act_high = float(env.action_space.high[0])
        self.max_action = max(abs(self.act_low), abs(self.act_high))
        self.action_dim = int(np.prod(env.action_space.shape))
        scale_corr_per_dim = (
            0.0 if self.max_action == 1.0 else float(np.log(self.max_action))
        )
        self._scale_correction = torch.tensor(
            scale_corr_per_dim * self.action_dim,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1)

    # ========== Policy interaction ==========
    def select_action(self, state: Tensor) -> Tensor:
        if self.steps_done < self.learning_starts:
            self.steps_done += 1
            return self.select_random_action()

        state = state.float()
        with torch.no_grad():
            outs, *_ = self.net(state=state, actor_pass=True)

        if self.net.action_type == "continuous":
            mean, std = outs[0]
            log_std = (std + 1e-6).log().clamp(MIN_LOG_STD, MAX_LOG_STD)
            std = torch.exp(log_std)
            z = mean + std * torch.randn_like(mean) if self.net.training else mean
            action = torch.tanh(z) * self.max_action
            return action.detach()

        return outs.detach()

    def select_greedy_action(self, state: Tensor, eval: bool = False) -> Tensor:
        if eval:
            self.net.eval()
            self.net.training = False
        with torch.no_grad():
            a = self.select_action(state)
        if eval:
            self.net.train()
            self.net.training = True
        return a

    # ========== Training step ==========
    def optimize_model(self, time_step):
        if self.steps_done < self.learning_starts:
            return
        if len(self.experience_replay) < self.batch_size:
            return

        grad_updates = max(1, int(self.gradient_steps))
        for _ in range(grad_updates):
            transitions = self.get_transitions()
            actor_loss, critic_loss, alpha_loss = self.compute_losses(*transitions)
            self.update_parameters(actor_loss, critic_loss, alpha_loss)

    # ========== Losses ==========
    def compute_losses(
        self, state_batch, action_batch, next_state_batch, reward_batch, mask_batch
    ):
        # ---- Target backup y ----
        with torch.no_grad():
            # a' ~ pi(a|s')
            outs_next, *_ = self.net(state=next_state_batch, actor_pass=True)
            next_mean, next_std = outs_next[0]
            log_std = (next_std + 1e-6).log().clamp(MIN_LOG_STD, MAX_LOG_STD)
            next_std = torch.exp(log_std)
            next_z = next_mean + next_std * torch.randn_like(next_mean)
            next_action = torch.tanh(next_z) * self.max_action

            # log pi(a') with tanh & scale corrections
            log_prob_gauss = self._log_prob_gauss(next_z, next_mean, log_std)
            log_det = self._tanh_log_det_jac(next_z)
            log_prob_policy = log_prob_gauss - log_det - self._scale_correction

            # Q_target: tüm target başların ortalaması (stabil)
            q_targets = []
            sa_next = torch.cat([next_state_batch, next_action], dim=-1)
            for tc in self.net.target_critics:
                q_targets.append(tc(sa_next)[0])
            target_q_mean = torch.stack(q_targets, dim=1).mean(dim=1)

            alpha = torch.exp(self.log_alpha)
            y = reward_batch + mask_batch * self.gamma * (
                target_q_mean - alpha * log_prob_policy
            )
            y = y.float()

        # ---- Critic loss: tüm başlar MSE -> ortalama ----
        critic_losses = []
        sa = torch.cat([state_batch, action_batch], dim=-1)
        for c in self.net.critics:
            q = c(sa)[0]
            critic_losses.append(F.mse_loss(q, y))
        critic_loss = torch.stack(critic_losses).mean()
        self.writer.critic_losses.append(critic_loss.item())

        # ---- Actor loss ----
        outs, *_ = self.net(state=state_batch, actor_pass=True)
        mean, std = outs[0]
        log_std = (std + 1e-6).log().clamp(MIN_LOG_STD, MAX_LOG_STD)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(mean)
        action_sample = torch.tanh(z) * self.max_action

        log_prob_gauss = self._log_prob_gauss(z, mean, log_std)
        log_det = self._tanh_log_det_jac(z)
        log_prob_policy = log_prob_gauss - log_det - self._scale_correction

        # Q(s, a_pi): başların ortalaması
        sa_pi = torch.cat([state_batch, action_sample], dim=-1)
        q_heads = []
        for c in self.net.critics:
            q_heads.append(c(sa_pi)[0])
        q_pi_mean = torch.stack(q_heads, dim=1).mean(dim=1)

        alpha = torch.exp(self.log_alpha)
        actor_loss = (alpha * log_prob_policy - q_pi_mean).mean()
        self.writer.actor_losses.append(actor_loss.item())

        # ---- Alpha loss ----
        alpha_loss = -(
            self.log_alpha * (log_prob_policy + self.target_entropy).detach()
        ).mean()
        self.writer.alpha_losses.append(alpha_loss.item())

        return actor_loss, critic_loss, alpha_loss

    # ========== Apply grads & soft-update ==========
    def update_parameters(
        self,
        actor_loss: Tensor,
        critic_loss: Tensor,
        alpha_loss: Tensor,
    ):
        # Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.gradient_clipping_max_norm:
            torch.nn.utils.clip_grad_norm_(
                self.net.actor.parameters(), self.gradient_clipping_max_norm
            )
        self.actor_optimizer.step()

        # Critics (tüm başlar tek optimizer)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.gradient_clipping_max_norm:
            for c in self.net.critics:
                torch.nn.utils.clip_grad_norm_(
                    c.parameters(), self.gradient_clipping_max_norm
                )
        self.critic_optimizer.step()

        # Alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update targets (tüm başlar)
        for c, tc in zip(self.net.critics, self.net.target_critics):
            for p, tp in zip(c.parameters(), tc.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ========== Utils ==========
    def get_transitions(self):
        transitions: Transition = self.experience_replay.sample()
        state_batch = transitions.state.squeeze(1)
        next_state_batch = transitions.next_state.squeeze(1)
        action_batch = transitions.action.squeeze(1)
        reward_batch = transitions.reward.squeeze(1)
        done_batch = transitions.done.squeeze(1).int()
        mask_batch = (1 - done_batch).float()
        return (state_batch, action_batch, next_state_batch, reward_batch, mask_batch)

    def decode_gym_action(self, nn_action_values):
        return super().decode_gym_action(nn_action_values)

    def _tanh_log_det_jac(self, z: torch.Tensor) -> torch.Tensor:
        # log(1 - tanh(z)^2) = -2 * (softplus(2z) - z - ln 2)
        return (-2.0 * (F.softplus(2.0 * z) - z - LOG_TWO)).sum(dim=-1, keepdim=True)

    def _log_prob_gauss(self, z, mean, log_std):
        return (
            -0.5
            * (((z - mean) / torch.exp(log_std)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        ).sum(dim=-1, keepdim=True)

    # ========== HER Support ==========
    def push_transition(self, transition: Transition):
        """
        Push a transition to the replay buffer.
        For standard replay buffers, this just calls push().
        For HER, use push_her_transition() instead.

        Args:
            transition: A Transition namedtuple
        """
        self.experience_replay.push(transition)

    def push_her_transition(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward: Tensor,
        done: bool,
        achieved_goal: Tensor,
        desired_goal: Tensor,
        next_achieved_goal: Tensor,
        info: dict = None,
    ):
        """
        Push a goal-conditioned transition to the HER buffer.

        This method should be used when experience_replay_type='her'.
        It stores the transition with goal information for hindsight relabeling.

        Args:
            state: Current observation
            action: Action taken
            next_state: Next observation
            reward: Reward received
            done: Whether episode ended
            achieved_goal: Goal achieved in current state
            desired_goal: Goal we wanted to achieve
            next_achieved_goal: Goal achieved in next state
            info: Additional info dict (used for timeout detection)
        """
        if not self.use_her:
            # Fallback to standard transition
            transition = Transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
            )
            self.experience_replay.push(transition)
            return

        # Create goal-conditioned transition
        her_transition = GoalConditionedTransition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            next_achieved_goal=next_achieved_goal,
            info=info if info is not None else {},
        )
        self.experience_replay.push(her_transition)

    def end_her_episode(self):
        """
        Manually signal end of episode for HER buffer processing.

        This triggers the hindsight relabeling for the collected episode.
        Call this when an episode ends due to timeout rather than terminal state.
        """
        if self.use_her and hasattr(self.experience_replay, "end_episode"):
            self.experience_replay.end_episode()
