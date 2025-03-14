from stable_baselines3 import SAC
import gymnasium as gym
import continuous_maze_env


def main():
    env = gym.make(
        "ContinuousMaze-v0", level="level_one", max_steps=600, random_start=True
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
    )

    model.learn(total_timesteps=1_000_000)


if __name__ == "__main__":
    main()
