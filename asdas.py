#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader, TensorDataset
import continuous_maze_env


#############################################
# Helper distribution for MultiDiscrete
#############################################
class MultiCategorical:
    """
    Wrap a list of independent Categorical distributions (one for each discrete action)
    and provide a unified interface.
    """

    def __init__(self, dists):
        self.dists = dists

    def sample(self):
        # Sample from each distribution and stack into one tensor.
        samples = [d.sample() for d in self.dists]
        # Assume each sample has shape (batch_size,); stack along last dim.
        return torch.stack(samples, dim=-1)

    def log_prob(self, actions):
        # Expect actions to be a tensor of shape (..., num_discrete)
        # Compute the log_prob of each component and sum them.
        log_probs = [d.log_prob(actions[..., i]) for i, d in enumerate(self.dists)]
        return sum(log_probs)

    def entropy(self):
        entropies = [d.entropy() for d in self.dists]
        return sum(entropies)


#############################################
# Generic Actor-Critic for PPO
#############################################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.action_space = action_space

        # Build the critic network (common to all cases)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Determine action type and build actor accordingly.
        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
            self.is_discrete = False
            self.is_multidiscrete = False
            # Flatten the Box action shape (e.g. for multidimensional continuous)
            self.action_dim = int(np.prod(action_space.shape))
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, self.action_dim),
            )
            # Use a state-independent log_std parameter (one per action dimension)
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
            self.is_discrete = True
            self.is_multidiscrete = False
            self.action_dim = action_space.n
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, self.action_dim),
            )
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.is_continuous = False
            self.is_discrete = False
            self.is_multidiscrete = True
            # For MultiDiscrete, we build a common feature extractor then a head for each discrete variable.
            self.num_discrete = len(action_space.nvec)
            self.actor_base = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            # Create one head per discrete component.
            self.actor_heads = nn.ModuleList(
                [nn.Linear(hidden_size, n) for n in action_space.nvec]
            )
        else:
            raise NotImplementedError("Unsupported action space type.")

    def forward(self, state):
        """
        Given a batch of states, return a distribution over actions and the state value.
        """
        value = self.critic(state)
        if self.is_continuous:
            mean = self.actor(state)
            std = torch.exp(self.log_std)  # same std for all states
            dist = Normal(mean, std)
            return dist, value
        elif self.is_discrete:
            logits = self.actor(state)
            dist = Categorical(logits=logits)
            return dist, value
        elif self.is_multidiscrete:
            features = self.actor_base(state)
            dists = []
            for head in self.actor_heads:
                logits = head(features)
                dists.append(Categorical(logits=logits))
            dist = MultiCategorical(dists)
            return dist, value


#############################################
# GAE Advantage Calculation
#############################################
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    advantages = []
    gae = 0
    # Append last_value to bootstrap the value estimates
    values = values + [last_value]
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


#############################################
# PPO Update Function
#############################################
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
            dist, values = policy(batch_states)
            # Compute new log probabilities and entropy.
            if policy.is_continuous:
                new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1)
            else:
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

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


#############################################
# Main Training Loop
#############################################
def main():
    # Choose device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # --- Change the environment ID here to test with different types ---
    # Examples:
    #  - Continuous: "Pendulum-v1", "MountainCarContinuous-v0"
    #  - Discrete:   "CartPole-v1", "Acrobot-v1"
    #  - MultiDiscrete: (You may need a custom env or use gymnasium’s toy text envs)
    env_id = "Pendulum-v1"
    env = gym.make(env_id)
    # env = gym.make(
    #     "ContinuousMaze-v0", level="level_one", max_steps=500, random_start=True
    # )
    state_dim = np.prod(env.observation_space.shape)

    # Initialize the generic ActorCritic based on the env action space.
    policy = ActorCritic(state_dim, env.action_space, hidden_size=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Hyperparameters
    num_updates = 1000
    timesteps_per_batch = 2048
    ppo_epochs = 10
    mini_batch_size = 64
    gamma = 0.99
    lam = 0.95
    clip_param = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    total_steps = 0
    # Gymnasium reset returns (obs, info)
    state, info = env.reset()

    for update in range(1, num_updates + 1):
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        batch_steps = 0

        while batch_steps < timesteps_per_batch:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            dist, value = policy(state_tensor)
            action = dist.sample()
            # For continuous actions, ensure proper shape; for discrete, get scalar; for multidiscrete, get vector.
            if policy.is_continuous:
                log_prob = dist.log_prob(action).sum(axis=-1)
                action_np = action.cpu().numpy()[0]  # shape: (action_dim,)
            elif policy.is_discrete:
                log_prob = dist.log_prob(action)
                action_np = int(action.item())
            elif policy.is_multidiscrete:
                log_prob = dist.log_prob(action)
                action_np = action.cpu().numpy()[0]  # shape: (num_discrete,)
            else:
                raise NotImplementedError

            # Step the environment. Note: Gymnasium returns (obs, reward, terminated, truncated, info)
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

        # Compute bootstrap value for the last state.
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            _, last_value = policy(state_tensor)
            last_value = last_value.item()

        advs, rets = compute_gae(rewards, values, dones, last_value, gamma, lam)
        # Convert lists to tensors.
        states_tensor = torch.as_tensor(
            np.array(states), dtype=torch.float32, device=device
        )
        # For actions, use float for continuous and long for discrete/multidiscrete.
        if policy.is_continuous:
            actions_tensor = torch.as_tensor(
                np.array(actions), dtype=torch.float32, device=device
            )
        else:
            actions_tensor = torch.as_tensor(
                np.array(actions), dtype=torch.long, device=device
            )
        old_log_probs_tensor = torch.as_tensor(
            np.array(log_probs), dtype=torch.float32, device=device
        )
        returns_tensor = torch.as_tensor(
            np.array(rets), dtype=torch.float32, device=device
        )
        advantages_tensor = torch.as_tensor(
            np.array(advs), dtype=torch.float32, device=device
        )
        # Normalize advantages.
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        ppo_update(
            policy,
            optimizer,
            states_tensor,
            actions_tensor,
            old_log_probs_tensor,
            returns_tensor,
            advantages_tensor,
            clip_param,
            ppo_epochs,
            mini_batch_size,
            value_coef,
            entropy_coef,
        )

        # Evaluate the policy every 10 updates.
        if update % 10 == 0:
            eval_rewards = 0
            eval_episodes = 5
            for _ in range(eval_episodes):
                eval_state, _ = env.reset()
                done_eval = False
                episode_reward = 0
                while not done_eval:
                    eval_state_tensor = torch.as_tensor(
                        eval_state, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    dist, _ = policy(eval_state_tensor)
                    # For evaluation use the mean (for continuous) or the mode (for discrete)
                    if policy.is_continuous:
                        # For Normal, the mean is given by the actor network output.
                        with torch.no_grad():
                            mean = policy.actor(eval_state_tensor)
                        action = mean
                        action_np = action.cpu().numpy()[0]
                    elif policy.is_discrete:
                        logits = policy.actor(eval_state_tensor)
                        action = torch.argmax(logits, dim=-1)
                        action_np = int(action.item())
                    elif policy.is_multidiscrete:
                        features = policy.actor_base(eval_state_tensor)
                        actions_list = []
                        for head in policy.actor_heads:
                            logits = head(features)
                            actions_list.append(torch.argmax(logits, dim=-1))
                        action = torch.stack(actions_list, dim=-1)
                        action_np = action.cpu().numpy()[0]
                    else:
                        raise NotImplementedError

                    eval_state, reward, terminated, truncated, info = env.step(
                        action_np
                    )
                    done_eval = terminated or truncated
                    episode_reward += reward
                eval_rewards += episode_reward
            eval_rewards /= eval_episodes
            print(
                f"Update {update}, Total Steps {total_steps}, Eval Reward: {eval_rewards:.2f}"
            )


if __name__ == "__main__":
    main()
