import gymnasium as gym
import x_driving_env

from policy_based.cross_entropy import CrossEntropy
from value_based.dqn import VanillaDQN, Rainbow

from common.env_wrappers import make_atari_env


def main():
    # env = gym.make("CartPole-v0")
    env = make_atari_env("SkiingNoFrameskip-v4")
    env.reset()
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
        time_steps=1000000,
        learning_rate=3e-4,
        batch_size=64,
        gradient_steps=1,
        gamma=0.99,
        experience_replay_size=10000,
        render=False,
        exploration_percentage=50,
        target_update_frequency=1000,
        writing_period=1000,
        plot_train_sores=True,
        # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        normalize_observation=False,
        network_arch=[128, 128],
        network_type="cnn",
        device="cuda:0",
    )
    # model = Rainbow(
    #     env=env,
    #     time_steps=40000,
    #     learning_rate=3e-4,
    #     batch_size=64,
    #     gradient_steps=1,
    #     gamma=0.99,
    #     n_step=3,
    #     double_enabled=True,
    #     experience_replay_size=20000,
    #     render=False,
    #     exploration_percentage=10,
    #     target_update_frequency=1500,
    #     writing_period=1000,
    #     plot_train_sores=True,
    #     # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
    #     normalize_observation=False,
    #     network_arch=[128, 128],
    #     gradient_clipping_max_norm=0.3,
    # )
    model.train()
    model.save(folder="models", checkpoint="last")


if __name__ == "__main__":
    main()
