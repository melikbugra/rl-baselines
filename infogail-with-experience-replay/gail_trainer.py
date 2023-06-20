import gym
import torch
from torch import nn
import matplotlib.pyplot as plt

import numpy as np
import pickle
import random
import json
import simple_maze_env

from collections import deque
import torch.optim as optim

from neural_networks import Discriminator, Generator, Info
from replay_memory import ReplayMemory, Transition


class Trainer():
    def __init__(self, env, render, episodes, batch_size, learning_rate, gen_fc_num, gen_fc_neuron_nums, disc_fc_num, disc_fc_neuron_nums):
        self.env = env
        self.device = torch.device('cpu')
        self.state_size = np.prod(self.env.observation_space.shape)
        self.action_size = self.env.action_space.n
        self.cluster_size = 1

        self.render = render

        self.get_train_data()

        self.episodes = episodes
        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.memory = ReplayMemory(
            self.state_size + self.cluster_size, self.env.action_space.n, 10000, batch_size
        )

        self.trained = 0
        self.score_list = deque(maxlen=100)

        self.gen_loss_list = []
        self.disc_loss_list = []
        self.info_loss_list = []
        self.episode_scores = []

        self.generator = Generator(self.state_size + self.cluster_size, fc_num=gen_fc_num, fc_neuron_nums=gen_fc_neuron_nums, output_neurons=self.action_size)
        self.discriminator = Discriminator(self.state_size + self.action_size, fc_num=disc_fc_num, fc_neuron_nums=disc_fc_neuron_nums)
        self.info = Info(self.state_size + self.action_size, fc_num=disc_fc_num, fc_neuron_nums=disc_fc_neuron_nums)
        self.info.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.loss_func_discriminator = nn.BCELoss()
        self.loss_func_generator = nn.BCELoss()
        self.loss_func_info = nn.BCELoss()

        self.loss_discriminator = 0
        self.loss_generator = 0
        self.loss_info = 0

        self.gen_optimizer = optim.AdamW(self.generator.parameters(), lr=learning_rate, amsgrad=True)
        self.disc_optimizer = optim.AdamW(self.discriminator.parameters(), lr=learning_rate, amsgrad=True)
        self.info_optimizer = optim.AdamW(self.info.parameters(), lr=learning_rate, amsgrad=True)

        plt.ion()

    def plot_scores(self, show_result=False):
        plt.figure(1)
        scores = torch.tensor(self.episode_scores, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(scores.numpy())
        # Take 100 episode averages and plot them too
        if len(scores) >= 100:
            means = scores.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    def plot_losses(self, show_result=False):
        plt.figure(2)
        gen_loss = torch.tensor(self.gen_loss_list, dtype=torch.float)
        disc_loss = torch.tensor(self.disc_loss_list, dtype=torch.float)
        info_loss = torch.tensor(self.info_loss_list, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        if len(gen_loss) >= 100:
            means_gen = gen_loss.unfold(0, 100, 1).mean(1).view(-1)
            means_gen = torch.cat((torch.zeros(99), means_gen))
            plt.plot(means_gen.numpy())

            means_disc = disc_loss.unfold(0, 100, 1).mean(1).view(-1)
            means_disc = torch.cat((torch.zeros(99), means_disc))
            plt.plot(means_disc.numpy())

            means_info = info_loss.unfold(0, 100, 1).mean(1).view(-1)
            means_info = torch.cat((torch.zeros(99), means_info))
            plt.plot(means_info.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    def get_train_data(self):
        with open("expert_data/states_info.pickle", "rb") as pcl:
            states = pickle.load(pcl)

        with open("expert_data/actions_encoded_info.pickle", "rb") as pcl:
            actions = np.array(pickle.load(pcl))

        self.expert_len = int(len(states))
        print(self.expert_len)

        self.samples = np.append(states, actions.reshape(-1,self.action_size), axis=1)

    def optimize_models(self):
        # training
        if len(self.memory) < self.batch_size:
            return
        mini_batch_gen = self.memory.sample()
        info_labels = mini_batch_gen.state[0].squeeze(1)[:, self.state_size].float().view(-1, 1)
        states_batch_gen = mini_batch_gen.state[0].squeeze(1)
        actions_batch_gen = mini_batch_gen.action[0].squeeze(1)

        mini_batch_gen_for_disc = torch.cat((states_batch_gen[:, :self.state_size], actions_batch_gen), dim=1)
        mini_batch_gen_for_gen = states_batch_gen
        mini_batch_gen_labels = torch.zeros((len(mini_batch_gen_for_disc), 1), device=self.device)

        batch_indices = np.random.choice(len(self.samples), self.batch_size, replace=False)
        mini_batch_disc = torch.tensor(self.samples[batch_indices], device=self.device)
        mini_batch_disc_labels = torch.ones((len(mini_batch_disc), 1), device=self.device)

        all_samples = torch.cat((mini_batch_disc, mini_batch_gen_for_disc)).float()
        all_samples_labels = torch.cat(
            (mini_batch_disc_labels, mini_batch_gen_labels)
        )

        # Training the discriminator
        self.disc_optimizer.zero_grad()
        output_discriminator = self.discriminator(all_samples)
        
        self.loss_discriminator = self.loss_func_discriminator(output_discriminator, all_samples_labels)
        self.loss_discriminator.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), 100)
        self.disc_optimizer.step()

        # Training the info
        self.info.zero_grad()
        generated_actions = self.generator(mini_batch_gen_for_gen)
        generated_samples = torch.cat((mini_batch_gen_for_gen[:,:-self.cluster_size], generated_actions.view(self.batch_size, self.action_size)), dim=1)
        output_info = self.info(generated_samples)
        self.loss_info = self.loss_func_info(output_info, info_labels)
        self.loss_info.backward()
        self.info_optimizer.step()

        # Training the generator
        self.gen_optimizer.zero_grad()
        generated_actions = self.generator(mini_batch_gen_for_gen)
        
        generated_samples = torch.cat((mini_batch_gen_for_gen[:,:-self.cluster_size], generated_actions.view(self.batch_size, self.action_size)), dim=1)
        output_discriminator_generated = self.discriminator(generated_samples)
        output_info_generated = self.info(generated_samples)
        self.loss_generator = self.loss_func_generator(
            output_discriminator_generated, mini_batch_disc_labels
        ) + self.loss_func_info(output_info_generated, info_labels)*5


        self.loss_generator.backward()
        torch.nn.utils.clip_grad_value_(self.generator.parameters(), 100)
        self.gen_optimizer.step()

    def encode_action(self, act):
        init_act = [0 for i in range(self.action_size)]
        init_act [act] = 1
        encoded_action = torch.tensor(init_act, device=self.device).view(1, -1)

        return encoded_action
    
    def act(self, state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = self.generator(state).max(1)[1].view(1, 1)
            return action
        
    def save_model(self, meta_data):
        print("Saving model...")
        torch.save(self.discriminator.state_dict(), f"gail_model/gail_disc_model.ckpt")
        torch.save(self.generator.state_dict(), f"gail_model/gail_gen_model.ckpt")
        print("Saving model... - Finished.")

        print("Saving metadata...")
        with open(f"gail_model/meta_data.json", "w") as jsn:
            json.dump(meta_data, jsn)
        print("Saving metadata... - Finished.")

    def train(self):
        self.best_avg = -float('inf')
        for episode in range(1, self.episodes):
            # Initialize the environment and get it's state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
            info_label = torch.randint(0, 2, (1,1))
            state = torch.cat((state, info_label), dim=1)
            print(f"Episode: {episode}, Cluster: {int(info_label)}")
            score = 0

            while True:
                if self.render:
                    self.env.render()
                action = self.act(state)
                observation, reward, done , _ = self.env.step(action.item())
                score += reward
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
                next_state = torch.cat((next_state, info_label), dim=1)

                # Store the transition in memory
                transition = Transition(
                            state=state,
                            action=self.encode_action(action.item())
                        )

                self.memory.push(transition)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_models()

    
                if done:
                    self.episode_scores.append(score)
                    self.gen_loss_list.append(self.loss_generator)
                    self.disc_loss_list.append(self.loss_discriminator)
                    self.info_loss_list.append(self.loss_info)
                    self.plot_scores()
                    self.plot_losses()
                    print(f"Episode: {episode},\tScore: {score},\tMean of last a hundred episodes: {np.mean(self.episode_scores)}")

                    break
            if episode % 10 == 0 and episode != 0:
                # print(f"Episode: {e}, Mean of last a hundred episodes: {np.mean(self.score_list)}")
                print(f"\tLoss Disc.: {self.loss_discriminator}")
                print(f"\tLoss Gen.: {self.loss_generator}")
                print(f"\t\tLoss Info.: {self.loss_info}")

        meta_data = {
                    "trained": self.trained,
                    "last_a_hundred_episodes": list(self.score_list)
                }
        self.save_model(meta_data)

        self.plot_scores(show_result=True)
        self.plot_losses(show_result=True)
        plt.ioff()
        plt.show()

        return np.mean(self.score_list)
    
    def play(self, env, episodes, cluster):
        from time import sleep
        self.generator.eval()
        for e in range(episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
            # info_label = torch.ones(1,1)
            info_label = torch.tensor([[cluster]])
            state = torch.cat((state, info_label), dim=1)
            print(f"Episode: {e}, Cluster: {int(info_label)}")
            score = 0

            while True:
                env.render()
                sleep(1)
                # act
                action = self.act(state).item()
                # print(agent.epsilon)

                # step
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0).view(1, -1)
                next_state = torch.cat((next_state, info_label), dim=1)

                # update state
                state = next_state

                score += reward
                
                
                if done:
                    env.render()
                    sleep(1)
                    print(f"Episode: {e},\tScore: {score}")
                    break
    
if __name__ == "__main__":
    maze = [
            [9, 9, 9, 9, 9],
            [9, 1, 0, 0, 9],
            [9, 0, 9, 0, 9],
            [9, 0, 0, 0, 9],
            [9, 9, 9, 2, 9],
            [9, 9, 9, 9, 9],

        ]
    env = gym.make("SimpleMaze-v1", maze=maze, max_steps=500)
    
    trainer = Trainer(
                env=env, render=False, episodes=5000, batch_size=64, learning_rate=3e-4, 
                gen_fc_num=2, gen_fc_neuron_nums=[256, 256],
                disc_fc_num=2, disc_fc_neuron_nums=[256, 256])
    trainer.train()

    # trainer.get_model()
    # trainer.train(episodes)