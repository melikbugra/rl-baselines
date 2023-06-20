import gym
import pickle
# from tensorflow import keras
import argparse
import json
import numpy as np
import pickle
import simple_maze_env
from pynput import keyboard
from itertools import count
import threading

KEY_MAPPING = {
    keyboard.Key.up: 0,
    keyboard.Key.left: 1,
    keyboard.Key.down: 2,
    keyboard.Key.right: 3
}

pressed_key = None

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

def save_data(states_all, actions_all):
    with open("expert_data/states.pickle", "wb") as pcl:
        pickle.dump(states_all, pcl)

    with open("expert_data/actions_encoded.pickle", "wb") as pcl:
        pickle.dump(actions_all, pcl)


def collect_data(episodes, clone_data_by=10000):
    states_all = []
    actions_all = []

    # try:
    #     with open("expert_data/states.pickle", "rb") as pcl:
    #         states_all = pickle.load(pcl)
    #         print(f"Previous expert data found with lenght of {len(states_all)}")

    #     with open("expert_data/actions_encoded.pickle", "rb") as pcl:
    #         actions_all = pickle.load(pcl)
    #         print(f"Previous actions data found with lenght of {len(actions_all)}")
    # except FileNotFoundError as e:
    #     print("No previous states or actions data found, creating a new one")
    #     states_all = []
    #     actions_all = []

    maze = [
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 1, 0, 0, 0, 8, 0, 0, 0, 9],
            [9, 0, 9, 9, 0, 0, 0, 9, 0, 9],
            [9, 0, 0, 9, 9, 8, 9, 9, 0, 9],
            [9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
            [9, 0, 9, 0, 9, 9, 0, 9, 0, 9],
            [9, 0, 9, 0, 0, 0, 0, 9, 0, 9],
            [9, 0, 0, 9, 0, 9, 0, 9, 0, 9],
            [9, 8, 0, 0, 0, 8, 2, 9, 8, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
        ]

    env = gym.make("SimpleMaze-v1", maze=maze)
    global pressed_key
    for e in range(episodes):
        states = []
        actions = []
        state = env.reset()
        env.render()
        state = np.reshape(state, [1, np.prod(state.shape)])
        states.append(state[0]+[0])

        
        while True:
            wait_for_input()
            action = KEY_MAPPING[pressed_key]

            # print(action)
            actions.append(encode_action(action))

            # step
            next_state, reward, done, info = env.step(action)
            env.render()
            next_state = np.reshape(next_state, [1, np.prod(next_state.shape)])
            if not done:
                states.append(next_state[0]+[0])
    
            if done:
                states_all.extend(states)
                actions_all.extend(actions)
                actions.clear()
                states.clear()
                print(f"Episode: {e}")
        
    states_all = states_all*clone_data_by
    actions_all = actions_all*clone_data_by
    print("Saving data...")
    save_data(states_all, actions_all)
    env.close()

if __name__=="__main__":
    episodes = parse_args()

    def on_key_press(key):
        global pressed_key
        if key in KEY_MAPPING:
            if hasattr(key, 'char'):
                pressed_key = key.char
            # Check if the pressed key is a special key
            elif hasattr(key, 'name'):
                pressed_key = key.name


    def on_key_release(key):
        if key == keyboard.Key.esc:
            # Stop the listener
            return False
        
    def wait_for_input():
        global pressed_key
        try:
            with keyboard.Listener(on_press=on_key_press) as listener:
                listener.join()
        finally:
            return pressed_key
    
    collect_data(episodes, clone_data_by=10000)
        