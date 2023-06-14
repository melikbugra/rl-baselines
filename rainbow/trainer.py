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

from dqn.replay_memory import Transition
from dqn.agent import DQNAgent



class Trainer:
    def __init__(self, env_name: str, render: bool, episodes: int, batch_size: int, 
                 gamma: float, epsilon_start: float, epsilon_end: float, exploration_percentage: float, 
                 learning_rate: float, fc_num: int, fc_neuron_nums: list[int], tau: float = 0.005, device: str = "cpu") -> None:

        self.env_name = env_name
        self.render = render
        self.env = gym.make(self.env_name)
        self.state_size = np.prod(self.env.observation_space.shape)
        self.action_size = self.env.action_space.n

        # plt.ion()

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_percentage = exploration_percentage
        self.tau = tau
        self.learning_rate = learning_rate
        self.fc_num = fc_num
        self.fc_neuron_nums = fc_neuron_nums
        
        self.agent = DQNAgent(env=self.env, device=self.device, episodes=self.episodes, 
                              learning_rate=self.learning_rate, gamma=self.gamma, batch_size=self.batch_size, 
                              epsilon_start=self.epsilon_start, epsilon_end=self.epsilon_end, exploration_percentage=self.exploration_percentage,
                              fc_num=self.fc_num, fc_neuron_nums=self.fc_neuron_nums)

    def train(self, trial_num = -1):
        self.best_avg = -float('inf')
        for episode in range(self.episodes):
            # Initialize the environment and get it's state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
            score = 0

            for t in count():
                if self.render:
                    self.env.render()
                # We select greedy action because nosiy network provides exploration
                action = self.agent.select_greedy_action(state)
                observation, reward, done , _ = self.env.step(action.item())
                score += reward
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)

                # Store the transition in memory
                transition = Transition(state=state,
                        action=action,
                        next_state=next_state,
                        reward=reward,
                        done=done)

                # N-step transition
                if self.agent.use_n_step:
                    one_step_transition = self.agent.memory_n.push(transition)
                # 1-step transition
                else:
                    one_step_transition = transition

                # add a single step transition
                if one_step_transition:
                    self.agent.memory.push(one_step_transition)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.agent.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.agent.target_net.state_dict()
                policy_net_state_dict = self.agent.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.agent.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.agent.adaptive_e_greedy_and_beta()
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
        env_name="LunarLander-v2", render=False, episodes=500, batch_size=64, gamma=0.99, epsilon_start=1, epsilon_end=0.001, exploration_percentage=10, learning_rate=3e-4, 
        fc_num=2, fc_neuron_nums=[128,128], tau=0.005, device="cpu")
    trainer.train()

        
                