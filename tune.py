from utils.tuner import Tuner


import gymnasium as gym

from value_based.dqn import VanillaDQN


def main():
    env_name = "CartPole-v0"
    param_dicts = [
        # {"name": "learning_rate", "low": 1e-4, "high": 1e-3, "type": "float"},
        # {"name": "exploration_percentage", "low": 1, "high": 90, "type": "int"},
        # {"name": "gradient_steps", "low": 1, "high": 2, "type": "int"},
        # {"name": "gamma", "low": 0.9, "high": 0.99, "type": "float"},
        {
            "name": "learning_rate",
            "choices": [1e-4, 2e-4, 3e-4, 4e-4, 5e-4],
            "type": "categorical",
        },
        {
            "name": "exploration_percentage",
            "choices": [10, 20, 50, 70, 90],
            "type": "categorical",
        },
        {"name": "gradient_steps", "choices": [1, 2, 3], "type": "categorical"},
        {"name": "gamma", "choices": [0.9, 0.93, 0.95, 0.99], "type": "categorical"},
        {
            "name": "time_steps",
            "choices": [int(1e4), int(2e4), int(3e4), int(4e4), int(5e4)],
            "type": "categorical",
        },
        {
            "name": "network_arch",
            "choices": ["[128, 128]", "[256, 256]"],
            "type": "categorical",
        },
        {
            "name": "experience_replay_size",
            "choices": [int(1e4), int(5e4)],
            "type": "categorical",
        },
        {
            "name": "batch_size",
            "choices": [64, 128, 256],
            "type": "categorical",
        },
        {
            "name": "normalize_observation",
            "choices": [True, False],
            "type": "categorical",
        },
    ]

    tuner = Tuner(
        env_name,
        model_class=VanillaDQN,
        param_dicts=param_dicts,
        n_jobs=-1,
        storage="postgresql://optuna:optuna@optuna-db.melikbugraozcelik.com/optuna",
    )

    tuner.tune()


if __name__ == "__main__":
    main()
