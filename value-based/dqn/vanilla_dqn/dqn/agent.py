import gym
import numpy as np
import torch
from dqn.replay_memory import ReplayMemory
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
    def __init__(self, env: gym.Env, device: torch.device, episodes: int, learning_rate: float, gamma: float, batch_size: int, epsilon_start: float, epsilon_end: float, exploration_percentage: float, fc_num: int, fc_neuron_nums: list[int]):
        self.env = env
        self.state_size = np.prod(self.env.observation_space.shape)
        self.action_size = self.env.action_space.n

        self.device = device
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.epsilon_decay = (self.epsilon - self.epsilon_end)*100/(self.episodes*(exploration_percentage + 1))

        self.policy_net: MLP = MLP(input_neurons=self.state_size, fc_num=2, fc_neuron_nums=[128, 128], output_neurons=self.action_size).to(self.device)
        self.target_net: MLP = MLP(input_neurons=self.state_size, fc_num=2, fc_neuron_nums=[128, 128], output_neurons=self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(self.state_size, 1, 10000, self.batch_size)

        self.steps_done = 0
        self.episode_scores = []

    def select_action(self, state):
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
        plt.figure("Vanilla DQN")
        scores = torch.tensor(self.episode_scores, dtype=torch.float)
        if show_result:
            plt.title('Vanilla DQN Result')
        else:
            plt.clf()
            plt.title('Training Vanilla DQN...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(scores.numpy(), color="blue")
        # Take 100 episode averages and plot them too
        if len(scores) >= 100:
            means = scores.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color="red")

        plt.pause(0.001)  # pause a bit so that plots are updated

    def adaptive_e_greedy(self):
        if self.epsilon > self.epsilon_end + 0.00001:
            self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions: Transition = self.memory.sample()
        state_batch = transitions.state[0].squeeze(1)
        next_state_batch = transitions.next_state[0].squeeze(1)
        action_batch = transitions.action[0].squeeze(1)
        reward_batch = transitions.reward[0].squeeze(1)
        done_batch = transitions.done.squeeze(1).int()

        mask = 1 - done_batch

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma * mask.squeeze(1)) + reward_batch.squeeze(1)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_model(self, env_name: str, checkpoint = ""):
        torch.save(self.policy_net.state_dict(), f"dqn/{env_name}_{checkpoint}.ckpt")

    def evaluate(self, env_name: str, episodes: int):
        env = gym.make(env_name)
        scores = []
        for episode in range(episodes):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
            score = 0

            while True:
                env.render()
                
                action = self.select_greedy_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                score += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)

                # Move to the next state
                state = next_state

                if done:
                    scores.append(score)
                    # print(f"Episode score: {score}")
                    sleep(1)
                    break

        return np.mean(scores)
