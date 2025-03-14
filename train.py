import gymnasium as gym

import continuous_maze_env
from rl_baselines.policy_based.cross_entropy import CrossEntropy
from rl_baselines.value_based.dqn import VanillaDQN, Rainbow
from rl_baselines.policy_based.reinforce import REINFORCE
from rl_baselines.policy_based.a2c import A2C
from rl_baselines.policy_based.ppo.ppo import PPO
from rl_baselines.policy_based.sac.sac import SAC

from rl_baselines.common.env_wrappers import make_atari_env, make_box2d_viz_env


def main():
    # env = gym.make("MountainCarContinuous-v0")
    # env = gym.make("Pendulum-v1")
    # env = gym.make("CartPole-v0")
    # env = gym.make(
    #     "ContinuousMaze-v0", level="level_one", max_steps=250, random_start=False
    # )
    env = make_box2d_viz_env("CarRacing-v2")
    # env = make_box2d_viz_env(
    #     "ContinuousMazeViz-v0", level="level_one", max_steps=259, random_start=True
    # )
    # env = make_atari_env("PongNoFrameskip-v4")
    # env = make_box2d_viz_env("CarRacing-v2")
    # env = make_atari_env(
    #     "WorldsHardestGame-v0", fire_reset=False
    # )  # TODO: Implement make custom env

    env.reset()

    model = SAC(
        env=env,
        time_steps=1_000_000,
        experience_replay_type="er",
        learning_rate=3e-4,
        network_type="cnn",
        # network_arch=[128, 128],
        render=False,
        device="cpu",
        plot_train_sores=True,
        writing_period=1000,
        tau=0.005,
        gamma=0.99,
        batch_size=256,
        # experience_replay_size=20000,
        target_entropy=-1.0,
        learning_starts=1000,
        # eval_env_kwargs={"level": "level_one", "max_steps": 250, "random_start": False},
    )

    # model = PPO(
    #     env=env,
    #     time_steps=1_000_000,
    #     learning_rate=3e-4,
    #     # network_arch=[64],
    #     network_type="actor_critic_cnn",
    #     device="cuda:0",
    #     writing_period=10000,
    #     plot_train_sores=True,
    #     render_eval=False,
    #     # mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
    #     log_model=False,
    #     n_epochs=10,
    #     batch_size=64,
    #     memory_size=2**5,
    #     gamma=0.99,
    #     clip_range=0.2,
    #     gae_lambda=0.95,
    #     entropy_coef=0.01,
    #     value_coef=0.5,
    #     render=False,
    #     # gradient_clipping_max_norm=1,
    #     normalize_observation=False,
    #     eval_env_kwargs={"level": "level_one", "max_steps": 500, "random_start": True},
    # )

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
