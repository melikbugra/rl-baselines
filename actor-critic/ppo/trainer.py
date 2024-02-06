import gym
import matplotlib.pyplot as plt
from itertools import count
import numpy as np

import torch

from ppo.replay_memory import Transition
from ppo.agent import PPOAgent

import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

class Trainer:
    def __init__(self, env_name: str, render: bool, episodes: int, batch_size: int, 
                 alpha: float, gamma: float, policy_clip: float, fc_neuron_nums: list[int], n_epochs: int, gae_lambda: float, device: str = "cpu") -> None:

        self.env_name = env_name
        self.render = render
        self.env = gym.make(self.env_name)
        self.device = device

        plt.ion()

        self.episodes = episodes
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.alpha = alpha
        self.gamma = gamma
        self.fc_neuron_nums = fc_neuron_nums
        self.n_epochs = n_epochs
        
        self.agent = PPOAgent(env=self.env, device=self.device, episodes=self.episodes, policy_clip=policy_clip,
                              alpha=self.alpha, gamma=self.gamma, batch_size=self.batch_size, gae_lambda=gae_lambda,
                              fc_neuron_nums=fc_neuron_nums, n_epochs=self.n_epochs)

    def train(self, trial_num = -1):
        self.best_avg = -float('inf')
        n_steps = 0
        for episode in range(self.episodes):
            # Initialize the environment and get it's state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
            score = 0
            done = False
            while not done:
                if episode > 1000:
                    self.env.render()
                action, prob, value = self.agent.select_action(state)
                next_state, reward, done , _ = self.env.step(action)
                n_steps += 1
                score += reward
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)

                self.agent.remember(state=state, action=action, prob=prob, value=value, reward=reward, done=done)

                if n_steps % self.agent.memory.max_size == 0:
                    self.agent.optimize_model()

                state = next_state
                
                if done:
                    self.agent.episode_scores.append(score)
                    self.agent.plot_scores()
                    avg = np.mean(self.agent.episode_scores[-100:])
                    print(f"Episode: {episode},\tScore: {score},\tMean of last 100: {avg}")
                    if avg >= self.best_avg:
                        # print("Best avg achieved, saving model...")
                        self.best_avg = avg
                        self.agent.save_model(self.env_name)
                    break
        self.agent.save_model(env_name=self.env_name, checkpoint="latest")

        self.agent.plot_scores(show_result=True)
        plt.ioff()
        plt.show()

        return np.mean(self.agent.episode_scores)

if __name__ == "__main__":
    trainer = Trainer(
        env_name="CartPole-v0", render=True, episodes=1100, batch_size=10,
        alpha=1e-4, gamma=0.99, policy_clip=0.2, fc_neuron_nums=[128,128], device="cpu", n_epochs=4, gae_lambda=0.95)
    trainer.train()

        
                