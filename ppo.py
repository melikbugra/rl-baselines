import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPOActorCritic, self).__init__()
        # Ortak Katman
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Politika (Actor) Katmanı
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Değer (Critic) Katmanı
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value


def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(
    policy_net,
    optimizer,
    states,
    actions,
    log_probs_old,
    returns,
    advantages,
    clip_epsilon=0.2,
    epochs=4,
    batch_size=64,
):
    states = torch.stack(states)
    actions = torch.tensor(actions)
    log_probs_old = torch.stack(log_probs_old).detach()
    returns = torch.tensor(returns).detach()
    advantages = torch.tensor(advantages).detach()
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )  # Normalize advantages

    dataset = torch.utils.data.TensorDataset(
        states, actions, log_probs_old, returns, advantages
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for _ in range(epochs):
        for batch in dataloader:
            (
                state_batch,
                action_batch,
                log_prob_old_batch,
                return_batch,
                advantage_batch,
            ) = batch
            policy_logits, value = policy_net(state_batch)
            dist = Categorical(torch.softmax(policy_logits, dim=-1))
            log_prob = dist.log_prob(action_batch)

            ratio = torch.exp(log_prob - log_prob_old_batch)
            surr1 = ratio * advantage_batch
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                * advantage_batch
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(value.squeeze(), return_batch)

            loss = (
                policy_loss + 0.5 * value_loss
            )  # 0.5, değer kaybının ağırlığını belirler

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PPOActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)

    max_episodes = 1000
    max_steps = 500
    gamma = 0.99
    lam = 0.95
    update_timestep = 2000
    clip_epsilon = 0.2
    epochs = 4
    batch_size = 64

    timestep = 0
    memory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "values": []}
    episode_rewards = []

    for episode in range(1, max_episodes + 1):
        state, info = env.reset()
        ep_reward = 0

        for step in range(max_steps):
            state_tensor = torch.tensor([state], dtype=torch.float32)
            policy_logits, value = policy_net(state_tensor)
            dist = Categorical(torch.softmax(policy_logits, dim=-1))
            action = dist.sample()

            next_state, reward, done, truncated, info = env.step(action.item())

            memory["states"].append(state_tensor)
            memory["actions"].append(action)
            memory["log_probs"].append(dist.log_prob(action))
            memory["rewards"].append(reward)
            memory["values"].append(value.item())

            state = next_state
            ep_reward += reward
            timestep += 1

            if timestep >= update_timestep:
                advantages, returns = compute_advantages(
                    memory["rewards"], memory["values"], gamma, lam
                )
                ppo_update(
                    policy_net,
                    optimizer,
                    memory["states"],
                    memory["actions"],
                    memory["log_probs"],
                    returns,
                    advantages,
                    clip_epsilon,
                    epochs,
                    batch_size,
                )
                memory = {
                    "states": [],
                    "actions": [],
                    "log_probs": [],
                    "rewards": [],
                    "values": [],
                }
                timestep = 0

            if done or truncated:
                episode_rewards.append(ep_reward)
                break

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
            if avg_reward >= env.spec.reward_threshold:
                print(
                    f"Solved! Average reward: {avg_reward:.2f} >= {env.spec.reward_threshold}"
                )
                break

    env.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
