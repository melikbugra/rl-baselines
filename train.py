import gymnasium as gym

import continuous_maze_env
from rl_baselines.policy_based.cross_entropy import CrossEntropy
from rl_baselines.value_based.dqn import VanillaDQN, Rainbow
from rl_baselines.policy_based.reinforce import REINFORCE
from rl_baselines.policy_based.a2c import A2C
from rl_baselines.policy_based.ppo.ppo import PPO

from rl_baselines.common.env_wrappers import make_atari_env, make_box2d_viz_env


def main():
    env = gym.make("CartPole-v0")
    # env = gym.make(
    #     "ContinuousMaze-v0", level="level_two", max_steps=2500, random_start=True
    # )
    # env = make_box2d_viz_env("CarRacing-v2", continuous=False)

    # env = make_atari_env("PongNoFrameskip-v4")
    # env = make_box2d_viz_env("CarRacing-v2", continuous=False)
    # env = make_atari_env(
    #     "WorldsHardestGame-v0", fire_reset=False
    # )  # TODO: Implement make custom env

    env.reset()

    model = PPO(
        env=env,
        time_steps=100_000,
        learning_rate=3e-4,
        network_arch=[128, 256, 128],
        network_type="mlp",
        device="cpu",
        writing_period=1000,
        plot_train_sores=True,
        render_eval=False,
        # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        log_model=False,
        n_epochs=4,
        batch_size=5,
        gamma=0.99,
        clip_range=0.2,
        gae_lambda=0.95,
        gradient_clipping_value=100,
        # eval_env_kwargs={"level": "level_two", "max_steps": 2500, "random_start": True},
    )

    # model = A2C(
    #     env=env,
    #     time_steps=1_000_000,
    #     learning_rate=3e-4,
    #     # network_arch=[128, 128],
    #     network_type="cnn",
    #     device="cuda:0",
    #     writing_period=100_000,
    #     plot_train_sores=True,
    #     n_step=10,
    #     # render_eval=True,
    #     # render=True,
    #     # mlflow_tracking_uri="https://mlflow.melikbugraozcelik.com/",
    #     log_model=False,
    #     # eval_env_kwargs={"level": "level_two", "max_steps": 2500, "random_start": True},
    # )
    # model = REINFORCE(
    #     env=env,
    #     episodes_to_train=1000,
    #     learning_rate=3e-4,
    #     gamma=0.99,
    #     render=False,
    #     normalize_observation=False,
    #     # mlflow_tracking_uri="https://mlflow.melikbugraozcelik.com/",
    #     network_arch=[128, 128],
    #     network_type="mlp",
    #     device="cpu",
    #     writing_period=100,
    #     plot_train_sores=True,
    # )
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
    #     gradient_steps=3,
    #     gamma=0.99,
    #     experience_replay_size=10000,
    #     render=False,
    #     exploration_percentage=5,
    #     target_update_frequency=1000,
    #     writing_period=1000,
    #     plot_train_sores=True,
    #     # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
    #     normalize_observation=False,
    #     network_arch=[128, 128],
    #     network_type="mlp",
    #     device="cpu",
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
