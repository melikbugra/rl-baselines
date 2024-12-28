import gymnasium as gym

import worlds_hardest_game_env

from policy_based.cross_entropy import CrossEntropy
from value_based.dqn import VanillaDQN, Rainbow
from policy_based.reinforce import REINFORCE

from common.env_wrappers import make_atari_env, make_box2d_viz_env


def main():
    env = gym.make("CartPole-v1")
    # env = make_atari_env("PongNoFrameskip-v4")
    # env = make_box2d_viz_env("CarRacing-v2", continuous=False)
    # env = make_atari_env(
    #     "WorldsHardestGame-v0", fire_reset=False
    # )  # TODO: Implement make custom env

    env.reset()
    model = REINFORCE(
        env=env,
        time_steps=100000,
        learning_rate=3e-3,
        gamma=0.99,
        episodes_to_train=16,
        render=False,
        normalize_observation=False,
        network_arch=[128],
        network_type="mlp",
        device="cpu",
        writing_period=1000,
        plot_train_sores=True,
    )
    # model = CrossEntropy(
    #     env=env,
    #     time_steps=80000,
    #     learning_rate=0.01,
    #     batch_size=256,
    #     render=False,
    #     percentile=20,
    #     writing_period=1000,
    #     plot_train_sores=True,
    #     # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
    #     normalize_observation=False,
    #     network_arch=[128, 128],
    #     network_type="cnn",
    #     device="cuda:0",
    # )
    # model = VanillaDQN(
    #     env=env,
    #     time_steps=10000,
    #     learning_rate=3e-4,
    #     batch_size=64,
    #     gradient_steps=1,
    #     gamma=0.99,
    #     experience_replay_size=10000,
    #     render=False,
    #     exploration_percentage=10,
    #     target_update_frequency=1000,
    #     writing_period=1000,
    #     plot_train_sores=True,
    #     # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
    #     normalize_observation=False,
    #     network_arch=[128, 128],
    #     network_type="cnn",
    #     device="cuda:0",
    # )
    # model = Rainbow(
    #     env=env,
    #     time_steps=1000000,
    #     learning_rate=3e-4,
    #     batch_size=32,
    #     gradient_steps=1,
    #     gamma=0.99,
    #     experience_replay_size=20000,
    #     render=True,
    #     exploration_percentage=1,
    #     target_update_frequency=1000,
    #     writing_period=1000,
    #     plot_train_sores=True,
    #     # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
    #     normalize_observation=False,
    #     network_arch=[128, 128],
    #     gradient_clipping_max_norm=1.0,
    #     network_type="cnn",
    #     device="cuda:0",
    #     n_step=3,
    #     double_enabled=True,
    #     noisy_enabled=True,
    #     experience_replay_type="per",
    #     env_seed=99,
    #     render_eval=True,
    # )
    model.train()
    model.save(folder="models", checkpoint="last")


if __name__ == "__main__":
    main()
