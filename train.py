import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("CartPole-v1")
    model = VanillaDQN(
        env=env,
        time_steps=100000,
        learning_rate=0.0008358872020105484,
        batch_size=64,
        gradient_steps=1,
        gamma=0.99,
        experience_replay_size=50000,
        render=False,
        exploration_percentage=20,
        writing_period=10000,
        plot_train_sores=True,
        # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        normalize_observation=False,
        network_arch=[128, 128],
    )
    model.train()
    model.save(folder="models", checkpoint="last")


if __name__ == "__main__":
    main()
