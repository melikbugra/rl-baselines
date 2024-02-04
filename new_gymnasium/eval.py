import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("xDriving-v0")
    model = VanillaDQN(env)
    model.load("models/xDriving-v0_cpu_998000.ckpt")
    model.evaluate(render=True)


if __name__ == "__main__":
    main()
