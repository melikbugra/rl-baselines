#!/usr/bin/env python3
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# Actor-Critic network with a Gaussian policy
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        # We'll use a state-independent log_std (initially 0)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        value = self.critic(state)
        return mean, std, value


# Compute advantages and returns using Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    advantages = []
    gae = 0
    # Append last_value to bootstrap the value estimate
    values = values + [last_value]
    for t in reversed(range(len(rewards))):
        # If the episode ended at step t, the next state is terminal (mask=0)
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


# PPO update: multiple epochs over mini-batches of collected trajectories
def ppo_update(
    policy,
    optimizer,
    states,
    actions,
    old_log_probs,
    returns,
    advantages,
    clip_param,
    ppo_epochs,
    mini_batch_size,
    value_coef,
    entropy_coef,
):
    dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
    for _ in range(ppo_epochs):
        for batch in loader:
            (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ) = batch
            mean, std, values = policy(batch_states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1)
            # Calculate probability ratio (new/old)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
                * batch_advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((batch_returns - values.squeeze(-1)) ** 2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the Gymnasium Pendulum environment
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Hyperparameters
    hidden_size = 64
    lr = 3e-4
    num_updates = 1000
    timesteps_per_batch = 2048
    ppo_epochs = 10
    mini_batch_size = 64
    gamma = 0.99
    lam = 0.95
    clip_param = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    policy = ActorCritic(state_dim, action_dim, hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    total_steps = 0
    # Gymnasium's reset returns (observation, info)
    state, info = env.reset()

    for update in range(1, num_updates + 1):
        states = []
        actions = []
        rewards = []
        dones = []  # Done flags (True if episode finished)
        log_probs = []
        values = []

        batch_steps = 0
        # Collect experience until we have timesteps_per_batch samples
        while batch_steps < timesteps_per_batch:
            state_tensor = torch.FloatTensor(state).to(device)
            mean, std, value = policy(state_tensor.unsqueeze(0))
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            action_np = action.cpu().numpy()[0]

            # Gymnasium step returns (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action_np)
            done_flag = terminated or truncated

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done_flag)
            log_probs.append(log_prob.detach().cpu().item())
            values.append(value.item())

            state = next_state
            batch_steps += 1
            total_steps += 1

            if done_flag:
                state, info = env.reset()

        # Compute bootstrap value for the last state
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            _, _, last_value = policy(state_tensor.unsqueeze(0))
            last_value = last_value.item()

        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(
            rewards, values, dones, last_value, gamma, lam
        )
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        old_log_probs = torch.FloatTensor(log_probs).to(device)

        # Normalize advantages for better stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO policy and value function update
        ppo_update(
            policy,
            optimizer,
            states,
            actions,
            old_log_probs,
            returns,
            advantages,
            clip_param,
            ppo_epochs,
            mini_batch_size,
            value_coef,
            entropy_coef,
        )

        # Evaluate the current policy every 10 updates
        if update % 10 == 0:
            eval_rewards = 0
            eval_episodes = 5
            for _ in range(eval_episodes):
                eval_state, _ = env.reset()
                done_eval = False
                episode_reward = 0
                while not done_eval:
                    eval_state_tensor = torch.FloatTensor(eval_state).to(device)
                    # Use the mean action (deterministic) for evaluation
                    mean, _, _ = policy(eval_state_tensor.unsqueeze(0))
                    action = mean.cpu().detach().numpy()[0]
                    eval_state, reward, terminated, truncated, info = env.step(action)
                    done_eval = terminated or truncated
                    episode_reward += reward
                eval_rewards += episode_reward
            eval_rewards /= eval_episodes
            print(
                f"Update {update}, Total Steps {total_steps}, Eval Reward: {eval_rewards:.2f}"
            )


if __name__ == "__main__":
    main()
