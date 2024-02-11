import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("xDriving-v0")
    model = VanillaDQN(
        env=env,
        time_steps=5000000,
        learning_rate=3e-4,
        batch_size=128,
        experience_replay_size=100000,
        render=False,
        exploration_percentage=50,
        writing_period=10000,
        plot_train_sores=True,
        mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        normalize_observation=False,
        network_arch=[256, 256],
    )
    model.train()
    model.save(folder="models", checkpoint="last")


if __name__ == "__main__":
    main()
