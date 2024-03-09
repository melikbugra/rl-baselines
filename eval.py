import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN, Rainbow
from policy_based.cross_entropy import CrossEntropy
from common.env_wrappers.atari_wrappers import make_atari_env


def main():
    # env = gym.make("CartPole-v0", render_mode="human")
    env = make_atari_env("SkiingNoFrameskip-v4", render_mode="human")
    # env = gym.make("PongNoFrameskip-v4", render_mode="human")
    # env.reset()
    # env.step(1)
    model = VanillaDQN(env, network_type="cnn")
    # model = Rainbow(env)
    model.load("models/SkiingNoFrameskip-v4_Vanilla-DQN_cuda:0_best_avg.ckpt")
    model.evaluate(render=True, print_episode_score=True)


if __name__ == "__main__":
    main()
