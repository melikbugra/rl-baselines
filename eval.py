import gymnasium as gym

import continuous_maze_env
from rl_baselines.policy_based.a2c import A2C
from rl_baselines.policy_based.ppo.ppo import PPO
from rl_baselines.policy_based.sac.sac import SAC
from rl_baselines.common.env_wrappers import make_atari_env, make_box2d_viz_env


def main():
    # env = gym.make(
    #     "ContinuousMaze-v0", level="level_two", max_steps=250, random_start=False
    # )
    # env = make_box2d_viz_env(
    #     "ContinuousMazeViz-v0", level="level_one", max_steps=500, random_start=True
    # )
    # env = gym.make("Pendulum-v1")
    # env = make_atari_env("ALE/MarioBros-v5", render_mode="human")
    # env = gym.make("PongNoFrameskip-v4", render_mode="human")
    env = make_box2d_viz_env("CarRacing-v2")
    # env.reset()
    # env.step(1)
    model = SAC(
        env,
        # eval_env_kwargs={"level": "level_one", "max_steps": 250, "random_start": False},
        experience_replay_size=512,
        device="cuda:0",
    )
    # model = Rainbow(env)
    model.load(folder="models", checkpoint="best_avg", eval_mode=True)
    model.evaluate(render=True, print_episode_score=True)
    env.close()


if __name__ == "__main__":
    main()
