import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("xDriving-v0")
    model = VanillaDQN(
        env=env,
        time_steps=50000,
        learning_rate=3e-4,
        render=False,
        exploration_percentage=0,
        writing_period=500,
        plot_train_sores=True,
        mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        normalize_observation=True,
    )
    model.train()


if __name__ == "__main__":
    main()
