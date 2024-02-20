import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("CartPole-v0")
    model = VanillaDQN(
        env=env,
        time_steps=50000,
        learning_rate=1e-3,
        batch_size=64,
        gradient_steps=1,
        gamma=0.95,
        experience_replay_size=20000,
        render=False,
        exploration_percentage=20,
        target_update_frequency=1,
        writing_period=1000,
        plot_train_sores=True,
        # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        normalize_observation=False,
        network_arch=[128, 128],
        tau=0.0005,
    )
    model.train()
    model.save(folder="models", checkpoint="last")


if __name__ == "__main__":
    main()
