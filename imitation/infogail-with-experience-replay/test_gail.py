import gym
import highway_env
import pickle
# from tensorflow import keras
import torch as T
from torch.autograd import Variable
import argparse
from gail_trainer import Trainer
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Tests the trained agents on y_env.',
                                     usage='python test_gail.py --episodes <int episodes>')
    parser.add_argument("--episodes", type=int, required=False, help="Specify the episode number (default=1).", default=10)
    parser.add_argument("--cluster", type=int, required=False, help="Specify the cluster (default=1).", default=1)
    args = parser.parse_args()
    return args.episodes, args.cluster


def test_gail(episodes, cluster):
    device = T.device('cpu')
    
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
                env=env, render=True, episodes=20000, batch_size=64, learning_rate=3e-4, 
                gen_fc_num=2, gen_fc_neuron_nums=[256, 256],
                disc_fc_num=2, disc_fc_neuron_nums=[256, 256])
    trainer.generator.load_state_dict(T.load("gail_model/gail_gen_model.ckpt"))
    trainer.play(env, episodes, cluster)
    

    

if __name__=="__main__":
    test_gail(*parse_args())
    
