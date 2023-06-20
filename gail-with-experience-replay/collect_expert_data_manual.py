import gym
import pickle
# from tensorflow import keras
import argparse
import json
import numpy as np
import pickle
import simple_maze_env
import keyboard

ACTIONS_ALL = {"UP": 0, "LEFT": 1, "DOWN": 2, "RIGHT": 3}

def parse_args():
    parser = argparse.ArgumentParser(description='Tests the trained agents on cartpole.',
                                     usage='python TestAgent.py -episodes <int episodes>')

    parser.add_argument("-episodes", type=int, required=False, help="Specify the episode number (default=1).", default=1)
    args = parser.parse_args()
    return args.episodes

def encode_action(act):
    init_act = [0, 0, 0, 0, 0]
    init_act [act] = 1

    return np.array(init_act)

def load_configs():
    with open('config-speed.json') as f:
        return json.load(f)

def save_data(states_all, actions_all):
    with open("expert_data/states_manual.pickle", "wb") as pcl:
        pickle.dump(states_all, pcl)

    with open("expert_data/actions_encoded_manual.pickle", "wb") as pcl:
        pickle.dump(actions_all, pcl)


def collect_data(episodes):
    states_all = []
    actions_all = []

    try:
        with open("expert_data/states_manual.pickle", "rb") as pcl:
            states_all = pickle.load(pcl)
            print(f"Previous expert data found with lenght of {len(states_all)/40}")

        with open("expert_data/actions_encoded_manual.pickle", "rb") as pcl:
            actions_all = pickle.load(pcl)
            print(f"Previous actions data found with lenght of {len(actions_all)}")
    except FileNotFoundError as e:
        print("No previous states or actions data found, creating a new one")
        states_all = []
        actions_all = []

    maze = [
            [9, 9, 9, 9, 9],
            [9, 1, 0, 0, 9],
            [9, 0, 9, 0, 9],
            [9, 0, 0, 0, 9],
            [9, 9, 9, 2, 9],
            [9, 9, 9, 9, 9],

        ]

    env = gym.make("SimpleMaze-v1", maze=maze)
    len1 = 0
    len2 = 0

    for e in range(episodes):
        states = []
        actions = []
        state = env.reset()
        state = np.reshape(state, [1, np.prod(state.shape)])
        states.append(state[0])

        score = 0
        while True:
            env.render()

            if keyboard.is_pressed("W"):
                action = ACTIONS_ALL["UP"]
            if keyboard.is_pressed("A"):
                action = ACTIONS_ALL["LEFT"]
            if keyboard.is_pressed("S"):
                action = ACTIONS_ALL["DOWN"]
            if keyboard.is_pressed("D"):
                action = ACTIONS_ALL["RIGHT"]

            print(action)
            actions.append(encode_action(action))
            # print(agent.epsilon)

            # step
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, np.prod(next_state.shape)])
            if not done:
                states.append(next_state[0])

            # update state
            state = next_state

            score += reward
     
            if done:
                states_all.extend(states)
                actions_all.extend(actions)
                actions.clear()
                states.clear()


                print(f"Episode: {e},\tScore: {score}")
                break
            if keyboard.is_pressed("X"):
                print("Saving data...")
                save_data(states_all, actions_all)


    print("Saving data...")
    save_data(states_all, actions_all)
    env.close()

if __name__=="__main__":
    episodes = parse_args()

    collect_data(episodes)
