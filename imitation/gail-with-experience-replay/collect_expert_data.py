import gym
import pickle
# from tensorflow import keras
import torch as T
from torch.autograd import Variable
import argparse
from expert_model.agent import DQNAgent
import json
import numpy as np
import pickle
import simple_maze_env


def parse_args():
    parser = argparse.ArgumentParser(description='Tests the trained agents on cartpole.',
                                     usage='python TestAgent.py -episodes <int episodes>')

    parser.add_argument("-episodes", type=int, required=False, help="Specify the episode number (default=1).", default=1)
    args = parser.parse_args()
    return args.episodes

def encode_action(act):
    init_act = [0, 0, 0, 0]
    init_act [act] = 1

    return np.array(init_act)

def load_configs():
    with open('config-speed.json') as f:
        return json.load(f)


def test_agent(episodes):
    device = T.device('cpu')
    # initialize environment
    
    maze = [
            [9, 9, 9, 9, 9],
            [9, 1, 0, 0, 9],
            [9, 0, 9, 0, 9],
            [9, 0, 0, 0, 9],
            [9, 9, 9, 2, 9],
            [9, 9, 9, 9, 9],
        ]
    env = gym.make("SimpleMaze-v1", maze=maze, max_steps=500)

    agent = DQNAgent(env=env, device=device, episodes=episodes, 
                              learning_rate=0, gamma=0, batch_size=0, 
                              epsilon_start=0, epsilon_end=0, exploration_percentage=0,
                              fc_num=2, fc_neuron_nums=[256,256])

    # Load saved model if exists
    try:
        agent.policy_net.load_state_dict(T.load("expert_model/SimpleMaze-v1_top.ckpt"))
        agent.policy_net.eval()
        print(f"Saved agent found")
    except (FileNotFoundError, OSError):
        print("No saved agent found, exiting...")
        exit()

    states_all = []
    actions_all = []

    for e in range(episodes):
        states = []
        actions = []
        state = env.reset()
        state = np.reshape(state, [1, np.prod(state.shape)])
        states.append(state[0])
        state = T.from_numpy(state)
        state = Variable(state).float().to(device)

        score = 0
        while True:
            # env.render()
            # act
            action = agent.select_greedy_action(state).item()
            actions.append(encode_action(action))
            # print(agent.epsilon)

            # step
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, np.prod(next_state.shape)])
            if not done:
                states.append(next_state[0])
            next_state = T.from_numpy(next_state)
            next_state = Variable(next_state).float().to(device)

            # update state
            state = next_state

            score += reward
     
            if done:

                states_all.extend(states)
                actions_all.extend(actions)
                actions.clear()
                states.clear()

                agent.episode_scores.append(score)
                print(f"Episode: {e},\tScore: {score}\tMean of last a hundred episodes: {np.mean(agent.episode_scores)}")
                break
        

    states = np.array(states)
    actions = np.array(actions)

    with open("expert_data/states.pickle", "wb") as pcl:
        pickle.dump(states_all, pcl)

    with open("expert_data/actions_encoded.pickle", "wb") as pcl:
        pickle.dump(actions_all, pcl)

    env.close()

if __name__=="__main__":
    episodes = parse_args()

    test_agent(episodes)
