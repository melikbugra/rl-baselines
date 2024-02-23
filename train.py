import gymnasium as gym
import x_driving_env

from policy_based.cross_entropy import CrossEntropy
from value_based.dqn import VanillaDQN


def main():
    env = gym.make("LunarLander-v2")
    # model = CrossEntropy(
    #     env=env,
    #     time_steps=80000,
    #     learning_rate=0.01,
    #     batch_size=256,
    #     render=False,
    #     percentile=70,
    #     writing_period=1000,
    #     plot_train_sores=True,
    #     # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
    #     normalize_observation=False,
    #     network_arch=[128, 128],
    # )
    model = VanillaDQN(
        env=env,
        time_steps=20000,
        learning_rate=3e-4,
        batch_size=64,
        gradient_steps=1,
        gamma=0.99,
        experience_replay_size=10000,
        render=False,
        exploration_percentage=5,
        target_update_frequency=1000,
        writing_period=1000,
        plot_train_sores=True,
        # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        normalize_observation=False,
        network_arch=[128, 128],
        tau=1,
    )
    model.train()
    model.save(folder="models", checkpoint="last")


if __name__ == "__main__":
    main()
