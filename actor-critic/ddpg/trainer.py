import gym
import math
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from ddpg.replay_memory import Transition
from ddpg.agent import DDPGAgent



class Trainer:
    def __init__(self, env_name: str, render: bool, episodes: int, batch_size: int, 
                 gamma: float,
                 alpha: float, beta: float, fc_neuron_nums: list[int], tau: float = 0.005, device: str = "cpu") -> None:

        self.env_name = env_name
        self.render = render
        self.env = gym.make(self.env_name)
        self.device = device

        plt.ion()

        self.episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.fc_neuron_nums = fc_neuron_nums
        
        self.agent = DDPGAgent(env=self.env, device=self.device, episodes=self.episodes, 
                              alpha=self.alpha, beta=self.beta, gamma=self.gamma, batch_size=self.batch_size, tau=tau,
                              fc_neuron_nums=fc_neuron_nums)

    def train(self, trial_num = -1):
        self.best_avg = -float('inf')
        for episode in range(self.episodes):
            # Initialize the environment and get it's state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
            score = 0
            self.agent.noise.reset()
            for t in count():
                if self.render:
                    self.env.render()
                action = self.agent.select_action(state)
                next_state, reward, done , _ = self.env.step(action)
                score += reward
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)

                self.agent.remember(state=state, action=torch.tensor([action], device=self.device, dtype=torch.float32), next_state=next_state, reward=reward, done=done)

                self.agent.optimize_model()

                state = next_state

                self.agent.update_network_parameters()
                
                if done:
                    self.agent.episode_scores.append(score)
                    self.agent.plot_scores()
                    avg = np.mean(self.agent.episode_scores[-100:])
                    print(f"Episode: {episode},\tScore: {score},\tMean of last 100: {avg}")
                    # if avg >= self.best_avg:
                    #     # print("Best avg achieved, saving model...")
                    #     self.best_avg = avg
                    #     self.agent.save_model(self.env_name)
                    break
        self.agent.save_model(env_name=self.env_name, checkpoint="latest")

        self.agent.plot_scores(show_result=True)
        plt.ioff()
        plt.show()

        return np.mean(self.agent.episode_scores)

if __name__ == "__main__":
    trainer = Trainer(
        env_name="LunarLanderContinuous-v2", render=False, episodes=1000, batch_size=64, gamma=0.99, 
        alpha=1e-4, beta=1e-3, fc_neuron_nums=[512,512], device="cuda:0")
    trainer.train()

        
                