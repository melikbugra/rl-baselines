import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("xDriving-v0", render_mode="human")
    model = VanillaDQN(env)
    model.load("models/xDriving-v0_cpu_best_avg.ckpt")
    model.evaluate(render=True, print_episode_score=True)


if __name__ == "__main__":
    main()
