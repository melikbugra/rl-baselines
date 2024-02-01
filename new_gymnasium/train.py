import gymnasium as gym

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("CartPole-v0")
    model = VanillaDQN(
        env=env,
        time_steps=50000,
        render=False,
        exploration_percentage=5,
        writing_period=500,
        plot_train_sores=True,
        mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
    )
    model.train()


if __name__ == "__main__":
    main()
