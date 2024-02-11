import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("CartPole-v0")
    model = VanillaDQN(
        env=env,
        time_steps=10000,
        learning_rate=1e-4,
        batch_size=128,
        experience_replay_size=10000,
        render=False,
        exploration_percentage=10,
        writing_period=100,
        plot_train_sores=True,
        mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        normalize_observation=False,
        network_arch=[256, 256],
    )
    model.train()
    model.save(folder="models", checkpoint="last")


if __name__ == "__main__":
    main()
