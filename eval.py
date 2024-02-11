import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("CartPole-v0")
    model = VanillaDQN(env)
    model.load("models/CartPole-v0_cpu_last.ckpt")
    model.evaluate(render=True)


if __name__ == "__main__":
    main()
