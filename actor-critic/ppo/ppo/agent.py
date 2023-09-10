import gym
import numpy as np
import torch
from ppo.replay_memory import ReplayMemory, Transition
from neural_networks.actor import Actor
from neural_networks.critic import Critic
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from time import sleep


class PPOAgent:
    def __init__(self, env: gym.Env, device: torch.device, episodes: int, alpha: float, gamma: float,
                 batch_size: int, gae_lambda: float, policy_clip: float, fc_neuron_nums: list[int], n_epochs: int):
        self.env = env
        self.state_size = np.prod(self.env.observation_space.shape)
        self.action_size = self.env.action_space.n

        self.device = device
        self.episodes = episodes
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs

        self.actor: Actor = Actor(input_neurons=self.state_size, fc_neuron_nums=fc_neuron_nums, output_neurons=self.action_size).to(self.device)
        self.critic: Critic = Critic(input_neurons=self.state_size, fc_neuron_nums=fc_neuron_nums).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)
        self.memory = ReplayMemory(self.state_size, 1, size=20, batch_size=self.batch_size, device=self.device)

        self.steps_done = 0
        self.episode_scores = []

    def select_action(self, state):
        self.actor.eval()
        self.critic.eval()
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        self.actor.train()
        self.critic.train()


        return action, prob, value
    
    def remember(self, state, action, prob, value, reward, done):
        transition = Transition(state=state,
                        action=action,
                        prob=prob,
                        value=value,
                        reward=reward,
                        done=done)
        self.memory.push(transition)

    def save_model(self, env_name: str, checkpoint = ""):
        torch.save(self.actor.state_dict(), f"trained_models/{env_name}_{checkpoint}_actor.ckpt")
        torch.save(self.critic.state_dict(), f"trained_models/{env_name}_{checkpoint}_critic.ckpt")

    def load_model(self, env_name: str, checkpoint = ""):
        self.actor.load_state_dict((f"trained_models/{env_name}_{checkpoint}_actor.ckpt"))
        self.critic.load_state_dict(torch.load()), f"trained_models/{env_name}_{checkpoint}_critic.ckpt"
        
    def plot_scores(self, show_result=False):
        plt.figure("PPO")
        scores = torch.tensor(self.episode_scores, dtype=torch.float)
        if show_result:
            plt.title('PPO Result')
        else:
            plt.clf()
            plt.title('Training PPO...')
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
        for _ in range(self.n_epochs):
            
            transitions, mini_batches = self.memory.sample()
            state_batch = transitions.state[0].squeeze()
            action_batch = transitions.action[0].squeeze()
            old_prob_batch = transitions.prob[0].squeeze()
            value_batch = transitions.value[0].squeeze()
            reward_batch = transitions.reward[0].squeeze()
            done_batch = transitions.done.squeeze().int()

            advantage = np.zeros(len(reward_batch), dtype=np.float32)

            for t in range(len(reward_batch)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_batch)-1):
                    a_t += discount*(reward_batch[k].item() + self.gamma*value_batch[k+1].item()*\
                            (1-done_batch[k].item()) - value_batch[k].item())
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)

            for mini_batch in mini_batches:
                states = state_batch[mini_batch]
                old_probs = old_prob_batch[mini_batch]
                actions = action_batch[mini_batch]

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[mini_batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[mini_batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[mini_batch] + value_batch[mini_batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                # torch.nn.utils.clip_grad_value_(self.actor.parameters(), 100)
                # torch.nn.utils.clip_grad_value_(self.critic.parameters(), 100)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear()


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
