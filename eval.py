import gymnasium as gym


from value_based.dqn import VanillaDQN, Rainbow
from policy_based.cross_entropy import CrossEntropy
from common.env_wrappers import make_atari_env, make_box2d_viz_env
from policy_based.reinforce import REINFORCE


def main():
    env = gym.make("Pendulum-v1", render_mode="human")
    # env = make_atari_env("ALE/MarioBros-v5", render_mode="human")
    # env = gym.make("PongNoFrameskip-v4", render_mode="human")
    # env = make_box2d_viz_env("CarRacing-v2", continuous=False, render_mode="human")
    # env.reset()
    # env.step(1)
    model = REINFORCE(env)
    # model = Rainbow(env)
    model.load("models/Pendulum-v1_REINFORCE_cpu_best_avg.ckpt")
    model.evaluate(render=True, print_episode_score=True)
    env.close()


if __name__ == "__main__":
    main()
