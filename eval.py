import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN, Rainbow
from policy_based.cross_entropy import CrossEntropy
from common.env_wrappers import make_atari_env, make_box2d_viz_env


def main():
    # env = gym.make("CartPole-v0", render_mode="human")
    # env = make_atari_env("ALE/MarioBros-v5", render_mode="human")
    # env = gym.make("PongNoFrameskip-v4", render_mode="human")
    env = make_box2d_viz_env("CarRacing-v2", continuous=False, render_mode="human")
    # env.reset()
    # env.step(1)
    model = Rainbow(env, network_type="cnn")
    # model = Rainbow(env)
    model.load("models/CarRacing-v2_Rainbow_cuda:0_best_avg.ckpt")
    model.evaluate(render=True, print_episode_score=True)


if __name__ == "__main__":
    main()
