import gymnasium as gym

import continuous_maze_env
from rl_baselines.policy_based.a2c import A2C
from rl_baselines.policy_based.pppppoooooo.ppo import PPO
from rl_baselines.common.env_wrappers import make_atari_env, make_box2d_viz_env


def main():
    env = make_box2d_viz_env(
        "ContinuousMaze-v0", level="level_one", max_steps=500, random_start=True
    )
    # env = gym.make("Pendulum-v1")
    # env = make_atari_env("ALE/MarioBros-v5", render_mode="human")
    # env = gym.make("PongNoFrameskip-v4", render_mode="human")
    # env = make_box2d_viz_env("CarRacing-v2")
    # env.reset()
    # env.step(1)
    model = PPO(
        env,
        eval_env_kwargs={"level": "level_one", "max_steps": 500, "random_start": True},
    )
    # model = Rainbow(env)
    model.load("models/ContinuousMaze-v0_PPO_cuda:0_best_avg.ckpt")
    model.evaluate(render=True, print_episode_score=True)
    env.close()


if __name__ == "__main__":
    main()
