import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("LunarLander-v2")
    model = VanillaDQN(env)
    model.load("models/LunarLander-v2_cpu_best_avg.ckpt")
    model.evaluate(render=True)


if __name__ == "__main__":
    main()
