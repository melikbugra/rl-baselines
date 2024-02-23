import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN
from policy_based.cross_entropy import CrossEntropy


def main():
    env = gym.make("CartPole-v0", render_mode="human")
    # model = VanillaDQN(env)
    # model.load("models/CartPole-v0_cpu_last.ckpt")
    model = VanillaDQN(env)
    model.load("models/CartPole-v0_Vanilla-DQN_cpu_best_avg.ckpt")
    model.evaluate(render=True, print_episode_score=True)


if __name__ == "__main__":
    main()
