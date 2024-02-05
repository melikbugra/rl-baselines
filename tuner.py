from utils.tuner import Tuner


import gymnasium as gym

from value_based.dqn import VanillaDQN


def main():
    env = gym.make("LunarLander-v2")

    param_dicts = [
        {"name": "learning_rate", "low": 1e-4, "high": 1e-3, "type": "float"},
        {"name": "exploration_percentage", "low": 1, "high": 90, "type": "int"},
        {"name": "gradient_steps", "low": 1, "high": 10, "type": "int"},
        {"name": "gamma", "low": 0.9, "high": 0.99, "type": "float"},
        {
            "name": "time_steps",
            "choices": [int(1e5), int(3e5), int(5e5)],
            "type": "categorical",
        },
        {
            "name": "network_arch",
            "choices": ["[128, 128]", "[256, 256]"],
            "type": "categorical",
        },
        {
            "name": "experience_replay_size",
            "choices": [int(1e4), int(5e4), int(1e5)],
            "type": "categorical",
        },
        {
            "name": "batch_size",
            "choices": [64, 128, 256],
            "type": "categorical",
        },
    ]

    tuner = Tuner(
        env,
        model_class=VanillaDQN,
        param_dicts=param_dicts,
        mlflow_tracking_uri="http://mlflow.melikbugraozcelik.com/",
        n_jobs=4,
    )

    tuner.tune()


if __name__ == "__main__":
    main()
