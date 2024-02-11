import gymnasium as gym
import x_driving_env

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("xDriving-v0")
    model = VanillaDQN(
        env=env,
        time_steps=300000,
        learning_rate=0.00040594615247134503,
        batch_size=256,
        gradient_steps=3,
        gamma=0.93,
        experience_replay_size=10000,
        render=False,
        exploration_percentage=50,
        writing_period=10000,
        plot_train_sores=True,
        mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        normalize_observation=True,
        network_arch=[256, 256],
    )
    model.train()
    model.save(folder="models", checkpoint="last")


if __name__ == "__main__":
    main()
