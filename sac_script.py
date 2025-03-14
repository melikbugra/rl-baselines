import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Hyperparameters (default SAC)
ENV_NAME = "Pendulum-v1"
GAMMA = 0.99
TAU = 0.005  # target smoothing coefficient
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
BUFFER_CAPACITY = int(1e6)
BATCH_SIZE = 256
LEARNING_STARTS = 1000  # steps before learning begins (populate replay)
TOTAL_STEPS = 100000  # total environment steps for training
TARGET_ENTROPY = -1.0  # target entropy (–dim(A), Pendulum action dim = 1)
PRINT_FREQ = 5000  # Print average return every PRINT_FREQ steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Replay Buffer
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=int(1e6)):
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.max_size = capacity
        self.size = 0
        self.ptr = 0  # pointer to next insertion index

    def store(self, obs, act, rew, next_obs, done):
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.acts_buf[idx] = act
        self.rews_buf[idx] = rew
        self.done_buf[idx] = done
        # increment pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        # Convert to tensors
        obs_batch = torch.tensor(self.obs_buf[idxs], dtype=torch.float32).to(device)
        acts_batch = torch.tensor(self.acts_buf[idxs], dtype=torch.float32).to(device)
        rews_batch = (
            torch.tensor(self.rews_buf[idxs], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        next_obs_batch = torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32).to(
            device
        )
        done_batch = (
            torch.tensor(self.done_buf[idxs], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        return obs_batch, acts_batch, rews_batch, next_obs_batch, done_batch


# Actor Network (Gaussian policy with state-dependent std)
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super(Actor, self).__init__()
        hidden_dim = 256  # hidden layer size
        # Two-layer MLP
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, act_dim)
        self.log_std_linear = nn.Linear(hidden_dim, act_dim)
        # Action scaling
        self.max_action = max_action
        # Initialize weights for stability
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # Clamp log_std to avoid numerical issues
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, obs, deterministic=False):
        """Returns an action (np.array) for a given observation (np.array) using the policy."""
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
        mean, std = self.forward(obs_t)
        if deterministic:
            z = mean
        else:
            noise = torch.randn_like(mean)
            z = mean + std * noise
        action = torch.tanh(z) * self.max_action
        return action.cpu().detach().numpy()[0]


# Critic Network (Q-function)
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


# Initialize environment and agent
env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
# Action bounds
act_low = float(env.action_space.low[0])
act_high = float(env.action_space.high[0])
max_action = max(abs(act_low), abs(act_high))

# Create policy and critics
actor = Actor(obs_dim, act_dim, max_action).to(device)
critic1 = Critic(obs_dim, act_dim).to(device)
critic2 = Critic(obs_dim, act_dim).to(device)
# Create target critic networks
target_critic1 = Critic(obs_dim, act_dim).to(device)
target_critic2 = Critic(obs_dim, act_dim).to(device)
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())
target_critic1.eval()
target_critic2.eval()

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic1_optimizer = optim.Adam(critic1.parameters(), lr=CRITIC_LR)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=CRITIC_LR)
log_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=ALPHA_LR)

# Experience replay buffer
buffer = ReplayBuffer(obs_dim, act_dim, capacity=BUFFER_CAPACITY)

# Variables for tracking episode returns
episode_reward = 0
episode_returns = []  # list to store returns of episodes that finished
last_print_step = 0

# Training loop
obs, info = env.reset()
for t in range(1, TOTAL_STEPS + 1):
    # Select action: use random actions during warm-up
    if t < LEARNING_STARTS:
        action = env.action_space.sample()
    else:
        action = actor.get_action(obs, deterministic=False)

    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Store transition
    buffer.store(obs, action, reward, next_obs, float(done))
    episode_reward += reward

    obs = next_obs

    # Check for episode termination
    if done:
        episode_returns.append(episode_reward)
        obs, info = env.reset()
        episode_reward = 0

    # Print returns every PRINT_FREQ steps
    if t - last_print_step >= PRINT_FREQ and len(episode_returns) > 0:
        avg_return = np.mean(episode_returns)
        print(
            f"Step {t}: Average Return over {len(episode_returns)} episodes: {avg_return:.2f}"
        )
        episode_returns = []  # reset for next interval
        last_print_step = t

    # Update SAC agent after collecting enough experience
    if t >= LEARNING_STARTS:
        batch_obs, batch_act, batch_rew, batch_next_obs, batch_done = (
            buffer.sample_batch(BATCH_SIZE)
        )

        # Compute target Q values using target critics and current policy
        with torch.no_grad():
            next_mean, next_std = actor.forward(batch_next_obs)
            noise = torch.randn_like(next_mean)
            next_z = next_mean + next_std * noise
            next_action = torch.tanh(next_z) * max_action

            log_prob_gauss = -0.5 * (
                ((next_z - next_mean) / next_std) ** 2
                + 2 * torch.log(next_std)
                + np.log(2 * np.pi)
            )
            log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
            log_prob_policy = log_prob_gauss - (
                1 - torch.tanh(next_z) ** 2 + 1e-6
            ).log().sum(dim=-1, keepdim=True)

            target_q1 = target_critic1(batch_next_obs, next_action)
            target_q2 = target_critic2(batch_next_obs, next_action)
            target_min_q = torch.min(target_q1, target_q2)
            y = batch_rew + GAMMA * (1 - batch_done) * (
                target_min_q - torch.exp(log_alpha) * log_prob_policy
            )

        # Critic losses
        current_q1 = critic1(batch_obs, batch_act)
        current_q2 = critic2(batch_obs, batch_act)
        critic1_loss = nn.functional.mse_loss(current_q1, y)
        critic2_loss = nn.functional.mse_loss(current_q2, y)
        critic1_optimizer.zero_grad()
        critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        critic1_optimizer.step()
        critic2_optimizer.step()

        # Actor loss: maximize Q - α * log(pi)
        mean, std = actor.forward(batch_obs)
        noise = torch.randn_like(mean)
        z = mean + std * noise
        action_sample = torch.tanh(z) * max_action
        log_prob_gauss = -0.5 * (
            ((z - mean) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi)
        )
        log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
        log_prob_policy = log_prob_gauss - (1 - torch.tanh(z) ** 2 + 1e-6).log().sum(
            dim=-1, keepdim=True
        )

        q1_pi = critic1(batch_obs, action_sample)
        q2_pi = critic2(batch_obs, action_sample)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (torch.exp(log_alpha) * log_prob_policy - min_q_pi).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Entropy (alpha) loss
        alpha_loss = -(log_alpha * (log_prob_policy + TARGET_ENTROPY).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        # Soft-update target networks
        for param, target_param in zip(
            critic1.parameters(), target_critic1.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(
            critic2.parameters(), target_critic2.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# Save trained model weights
# torch.save(actor.state_dict(), "sac_actor.pth")
# torch.save(critic1.state_dict(), "sac_critic1.pth")
# torch.save(critic2.state_dict(), "sac_critic2.pth")
# print("Models saved to disk.")

# Evaluate the trained policy
eval_episodes = 5
avg_return = 0.0
actor.eval()
for ep in range(eval_episodes):
    obs, info = env.reset()
    done = False
    ep_return = 0.0
    while not done:
        action = actor.get_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_return += reward
    avg_return += ep_return
    print(f"Evaluation Episode {ep+1} return: {ep_return:.2f}")
avg_return /= eval_episodes
print(f"Average return over {eval_episodes} evaluation episodes: {avg_return:.2f}")
env.close()
