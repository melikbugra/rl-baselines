import gymnasium as gym

import continuous_maze_env
from rl_baselines.policy_based.a2c import A2C


def main():
    # env = gym.make(
    #     "ContinuousMaze-v0",
    #     level="level_two",
    #     max_steps=2500,
    #     random_start=True,
    #     render_mode="human",
    # )
    env = gym.make("MountainCarContinuous-v0")
    # env = make_atari_env("ALE/MarioBros-v5", render_mode="human")
    # env = gym.make("PongNoFrameskip-v4", render_mode="human")
    # env = make_box2d_viz_env("CarRacing-v2", continuous=False, render_mode="human")
    # env.reset()
    # env.step(1)
    model = A2C(
        env,
        # eval_env_kwargs={"level": "level_two", "max_steps": 2500, "random_start": True},
    )
    # model = Rainbow(env)
    model.load("models/MountainCarContinuous-v0_A2C_cpu_best_avg.ckpt")
    model.evaluate(render=True, print_episode_score=True)
    env.close()


if __name__ == "__main__":
    main()
