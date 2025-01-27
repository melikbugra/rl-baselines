import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# 1) Actor ve Critic için tek bir ağ kullanabilir veya iki ayrı ağ tanımlayabilirsiniz.
#    Burada ortak bir gövde + iki ayrı çıkış tanımı yapacağız (policy ve value).


class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128, action_dim=2):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # Policy (actor) çıktısı
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Değer fonksiyonu (critic) çıktısı
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        policy_logits = self.policy_head(x)  # actions için logits
        value = self.value_head(x)  # durumun değer tahmini
        return policy_logits, value


def select_action(model, state):
    """
    Politikadan eylem seçimi (olası eylemleri logit'lerle alıp softmax örnekliyoruz).
    """
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    policy_logits, value = model(state_tensor)

    # Categorical distribution
    probs = torch.softmax(policy_logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()

    return action.item(), dist.log_prob(action), value


def train_a2c(env, model, optimizer, gamma=0.99, n_steps=5, max_episodes=1000):
    """
    Tek environment üzerinden n-step A2C mantığı:
    - n adım (ya da done) bekleyip, sonra update.
    - Actor update => Advantage = R + gamma^n * V(next_state) - V(state)
    - Critic update => MSE( R + gamma^n * V(next_state), V(state) )
    """
    episode_rewards = []
    state = env.reset()[0]
    ep_reward = 0
    done = False

    for episode in range(max_episodes):

        # Tek episode'da n-step rollout döngüsü
        rollout = []  # (state, action, log_prob, value, reward)
        for t in range(n_steps):
            action, log_prob, value = select_action(model, state)
            next_state, reward, done, truncated, info = env.step(action)
            rollout.append((state, action, log_prob, value, reward))

            state = next_state
            ep_reward += reward

            if done or truncated:
                # Episode bitti, kaydet ve resetle
                episode_rewards.append(ep_reward)
                state = env.reset()[0]
                ep_reward = 0
                break

        # n-step (veya episode bitene kadarki) rollout üzerinden geri yayılım
        # Son durumun değerini tahmin edelim (bootstrap)
        if not done and not truncated:
            with torch.no_grad():
                _, next_value = model(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                )
                next_value = next_value.item()
        else:
            next_value = 0.0

        # Avantaj ve hedefler (returns) hesaplayalım
        returns = []
        advantages = []
        R = next_value

        for st, ac, lp, val, rew in reversed(rollout):
            R = rew + gamma * R
            advantage = R - val.item()
            returns.insert(0, R)
            advantages.insert(0, advantage)

        # Actor-Critic güncellemesi
        policy_loss = []
        value_loss = []

        for (st, ac, lp, val, rew), G, adv in zip(rollout, returns, advantages):
            # Actor loss = - log_prob(a) * advantage
            policy_loss.append(-lp * adv)

            # Critic loss = (G - V(s))^2
            value_loss.append(nn.functional.mse_loss(val, torch.tensor([[G]])))

        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(
                f"Episode {episode+1}/{max_episodes}, avg reward(last 50): {avg_reward:.2f}"
            )

    return episode_rewards


def main():
    env = gym.make("CartPole-v1")
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    rewards_history = train_a2c(
        env, model, optimizer, gamma=0.99, n_steps=5, max_episodes=100000
    )

    env.close()
    print("Training finished. Final average reward:", np.mean(rewards_history[-50:]))


if __name__ == "__main__":
    main()
