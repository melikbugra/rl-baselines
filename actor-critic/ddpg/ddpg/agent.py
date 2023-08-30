import gym
import numpy as np
import torch
from ddpg.replay_memory import ReplayMemory, Transition
from neural_networks.actor import Actor
from neural_networks.critic import Critic
from ddpg.noise import OUActionNoise
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from time import sleep


class DDPGAgent:
    def __init__(self, env: gym.Env, device: torch.device, episodes: int, alpha: float, 
                 beta: float, gamma: float, batch_size: int, tau: float, fc_neuron_nums: list[int],):
        self.env = env
        self.state_size = np.prod(self.env.observation_space.shape)
        self.action_size = self.env.action_space.shape[0]

        self.device = device
        self.episodes = episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.noise = OUActionNoise(mu=np.zeros(self.action_size))

        self.actor: Actor = Actor(input_neurons=self.state_size, fc_neuron_nums=fc_neuron_nums, output_neurons=self.action_size).to(self.device)
        self.critic: Critic = Critic(input_neurons=self.state_size, fc_neuron_nums=fc_neuron_nums, n_actions=self.action_size).to(self.device)

        self.target_actor: Actor = Actor(input_neurons=self.state_size, fc_neuron_nums=fc_neuron_nums, output_neurons=self.action_size).to(self.device)
        self.target_critic: Critic = Critic(input_neurons=self.state_size, fc_neuron_nums=fc_neuron_nums, n_actions=self.action_size).to(self.device)

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=alpha, amsgrad=True)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=beta, amsgrad=True, weight_decay=0.01)
        self.memory = ReplayMemory(self.state_size, self.action_size, 1000000, self.batch_size, device=self.device)

        self.steps_done = 0
        self.episode_scores = []

    def select_action(self, state):
        self.actor.eval()
        mu = self.actor(state)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float, device=self.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]
    
    def remember(self, state, action, next_state, reward, done):
        transition = Transition(state=state,
                        action=action,
                        next_state=next_state,
                        reward=reward,
                        done=done)
        self.memory.push(transition)

    def save_model(self, env_name: str, checkpoint = ""):
        torch.save(self.actor.state_dict(), f"trained_models/{env_name}_{checkpoint}_actor.ckpt")
        torch.save(self.critic.state_dict(), f"trained_models/{env_name}_{checkpoint}_critic.ckpt")
        torch.save(self.target_actor.state_dict(), f"trained_models/{env_name}_{checkpoint}_target_actor.ckpt")
        torch.save(self.target_critic.state_dict(), f"trained_models/{env_name}_{checkpoint}_target_critic.ckpt")

    def load_model(self, env_name: str, checkpoint = ""):
        self.actor.load_state_dict((f"trained_models/{env_name}_{checkpoint}_actor.ckpt"))
        self.critic.load_state_dict(torch.load()), f"trained_models/{env_name}_{checkpoint}_critic.ckpt"
        self.target_actor.load_state_dict(torch.load(f"trained_models/{env_name}_{checkpoint}_target_actor.ckpt"))
        self.target_critic.load_state_dict(torch.load(f"trained_models/{env_name}_{checkpoint}_target_critic.ckpt"))
        
    def plot_scores(self, show_result=False):
        plt.figure("DDPG")
        scores = torch.tensor(self.episode_scores, dtype=torch.float)
        if show_result:
            plt.title('DDPG Result')
        else:
            plt.clf()
            plt.title('Training DDPG...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(scores.numpy(), color="blue")
        # Take 100 episode averages and plot them too
        if len(scores) >= 100:
            means = scores.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color="red")

        plt.pause(0.001)  # pause a bit so that plots are updated

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions: Transition = self.memory.sample()
        state_batch = transitions.state[0].squeeze(1)
        next_state_batch = transitions.next_state[0].squeeze(1)
        action_batch = transitions.action[0].squeeze(1)
        reward_batch = transitions.reward[0].squeeze(1)
        done_batch = transitions.done.squeeze(1).int()

        target_actions = self.target_actor(state_batch)
        target_state_action_values = self.critic(next_state_batch, target_actions)
        state_action_values = self.critic(state_batch, action_batch)

        state_action_values[done_batch] = 0.0
        # state_action_values = state_action_values.view(-1)

        target = reward_batch + self.gamma*target_state_action_values

        self.critic_optimizer.zero_grad()

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()

        critic_loss = criterion(target, state_action_values)
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 100)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 100)
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_network_parameters(self):
        critic_state_dict = self.critic.state_dict()
        actor_state_dict = self.actor.state_dict()
        target_critic_state_dict = self.target_critic.state_dict()
        target_actor_state_dict = self.target_actor.state_dict()

        for key in critic_state_dict:
            critic_state_dict[key] = critic_state_dict[key].clone()*self.tau + target_critic_state_dict[key].clone()*(1-self.tau)

        for key in actor_state_dict:
            actor_state_dict[key] = actor_state_dict[key].clone()*self.tau + target_actor_state_dict[key].clone()*(1-self.tau)

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    # def evaluate(self, env_name: str, episodes: int):
    #     env = gym.make(env_name)
    #     scores = []
    #     for episode in range(episodes):
    #         state, info = env.reset()
    #         state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
    #         score = 0

    #         while True:
    #             env.render()
                
    #             action = self.select_greedy_action(state)
    #             observation, reward, terminated, truncated, _ = env.step(action.item())
    #             score += reward
    #             reward = torch.tensor([reward], device=self.device)
    #             done = terminated or truncated

    #             if terminated:
    #                 next_state = None
    #             else:
    #                 next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)

    #             # Move to the next state
    #             state = next_state

    #             if done:
    #                 scores.append(score)
    #                 # print(f"Episode score: {score}")
    #                 break

    #     return np.mean(scores)
