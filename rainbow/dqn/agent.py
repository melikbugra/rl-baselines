import gym
import numpy as np
import torch
from dqn.replay_memory import ReplayMemory, PrioritizedReplayMemory
from neural_networks.mlp import MLP
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
from dqn.replay_memory import Transition
import torch.nn as nn
from time import sleep
import pygame



class DQNAgent:
    def __init__(self, env: gym.Env, device: torch.device, episodes: int, 
                 learning_rate: float, gamma: float, batch_size: int, 
                 epsilon_start: float, epsilon_end: float, 
                 exploration_percentage: float, 
                 fc_num: int, fc_neuron_nums: list[int],
                 # PER
                 alpha: float=0.2, beta: float=0.6,
                 # Categorical
                 v_min: float = 0.0, v_max: float = 200.0, atom_size: int = 51,
                 # N-Step
                 n_step: int = 5):
        self.env = env
        self.state_size = np.prod(self.env.observation_space.shape)
        self.action_size = self.env.action_space.n

        self.device = device
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        # epsilon related parameters are read to keep consistency when calling the agent, but they are not used due to noisy network
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.epsilon_decay = (self.epsilon - self.epsilon_end)*100/(self.episodes*(exploration_percentage + 1))

        # PER related
        # Memory for 1 step
        self.alpha: float = alpha
        self.beta: float = beta
        self.beta_increase = (1 - self.beta)/(self.episodes)
        self.prior_eps: float = 1e-6
        self.memory = PrioritizedReplayMemory(self.state_size, 1, 10000, 64, self.alpha)

        # N-Step
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayMemory(
                self.state_size, 1, 10000, self.batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical related
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        self.policy_net: MLP = MLP(input_neurons=self.state_size, fc_num=2, fc_neuron_nums=[128, 128], output_neurons=self.action_size, atom_size=atom_size, support=self.support).to(self.device)
        self.target_net: MLP = MLP(input_neurons=self.state_size, fc_num=2, fc_neuron_nums=[128, 128], output_neurons=self.action_size, atom_size=atom_size, support=self.support).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        self.steps_done = 0
        self.episode_scores = []

    def select_e_greedy_action(self, state):
        sample = random.uniform(0, 1)
        self.steps_done += 1
        if sample > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
    def select_greedy_action(self, state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1)[1].view(1, 1)
        
    def plot_scores(self, show_result=False):
        plt.figure("Rainbow DQN")
        scores = torch.tensor(self.episode_scores, dtype=torch.float)
        if show_result:
            plt.title('Rainbow DQN Result')
        else:
            plt.clf()
            plt.title('Training Rainbow DQN...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(scores.numpy(), color="blue")
        # Take 100 episode averages and plot them too
        if len(scores) >= 100:
            means = scores.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color="red")

        plt.pause(0.001)  # pause a bit so that plots are updated

    def adaptive_e_greedy_and_beta(self):
        if self.epsilon > self.epsilon_end + 0.00001:
            self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end

        # PER: increase beta
        if self.beta < 1:
            self.beta += self.beta_increase
        if self.beta > 1:
            self.beta = 1

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        # PER
        transitions, weights, indices = self.memory.sample(self.beta)

        weights = torch.FloatTensor(
            weights.reshape(-1, 1)
        ).to(self.device)

        state_batch = transitions.state[0].squeeze(1)
        next_state_batch = transitions.next_state[0].squeeze(1)
        action_batch = transitions.action[0].squeeze(1)
        reward_batch = transitions.reward[0].squeeze(1)
        done_batch = transitions.done.squeeze(1).int()

        mask = 1 - done_batch

        # 1 - step learning loss
        elementwise_loss = self.compute_loss(state_batch, action_batch, next_state_batch, reward_batch, mask, self.gamma)

        # PER importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-Step Learning Loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step

            transitions = self.memory_n.sample_batch_from_idxs(indices)
            state_batch = transitions.state[0].squeeze(1)
            next_state_batch = transitions.next_state[0].squeeze(1)
            action_batch = transitions.action[0].squeeze(1)
            reward_batch = transitions.reward[0].squeeze(1)
            done_batch = transitions.done.squeeze(1).int()

            mask = 1 - done_batch

            elementwise_loss_n_loss = self.compute_loss(state_batch, action_batch, next_state_batch, reward_batch, mask, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # Noisy reset noise
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def compute_loss(self, state_batch, action_batch, next_state_batch, reward_batch, mask, gamma):
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # DDQN (polict_net is used instead of target)
            next_action = self.policy_net(next_state_batch).max(1)[1]
            next_dist = self.target_net.dist(next_state_batch)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward_batch + mask * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.policy_net.dist(state_batch)
        log_p = torch.log(dist[range(self.batch_size), action_batch.squeeze(1)])

        elementwise_loss  = -(proj_dist * log_p).sum(1)

        return elementwise_loss 

    def save_model(self, env_name: str, checkpoint = ""):
        torch.save(self.policy_net.state_dict(), f"dqn/{env_name}_{checkpoint}.ckpt")

    def evaluate(self, env_name: str, episodes: int, config):
        env = gym.make(env_name, config=config)
        scores = []
        for episode in range(episodes):
            self.policy_net.load_state_dict(torch.load(f"/home/melikozcelik/rl-projects/rainbow_gym/rainbow/dqn/highway-fast-v0_recent.ckpt"))
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
            score = 0

            while True:
                env.render()
                
                action = self.select_greedy_action(state)
                observation, reward, done, _ = env.step(action.item())
                score += reward
                reward = torch.tensor([reward], device=self.device)

                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)

                # Move to the next state
                state = next_state

                if done:
                    scores.append(score)
                    print(f"Episode score: {score}")
                    break

        return np.mean(scores)
