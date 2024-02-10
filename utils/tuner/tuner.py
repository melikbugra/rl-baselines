from datetime import datetime
import json
import os
from pathlib import Path
from typing import Type, Any

import gymnasium as gym
from gymnasium import Env
import optuna
from optuna.trial import BaseTrial
import numpy as np
from prettytable import PrettyTable

from utils.base_classes.base_algorithm import BaseAlgorithm


class Tuner:
    def __init__(
        self,
        env_name: str,
        model_class: Type[BaseAlgorithm],
        param_dicts: list[dict],
        sampler_seed: int = 42,
        n_trials: int = 100,
        n_jobs: int = 1,
        storage: str = None,
    ) -> None:
        self.param_dicts = param_dicts
        self.env_name: str = env_name
        self.model_class: Type[BaseAlgorithm] = model_class
        self.sampler_seed: int = sampler_seed
        self.n_trials: int = n_trials
        self.n_jobs: int = n_jobs
        self.storage: str = storage

        tuning_folder = Path("tuning")
        current_tuning_folder = (
            tuning_folder / f"{self.model_class.algo_name}_{self.env_name}"
        )
        os.makedirs("tuning", exist_ok=True)
        os.makedirs(current_tuning_folder, exist_ok=True)
        self.tried_params_file = current_tuning_folder / "tried_params_history.json"

        if os.path.exists(self.tried_params_file):
            self.tried_params_history = self.load_params_dict()
        else:
            self.tried_params_history = {
                "best_trial": -1,
                "trials": [],
                "params_history": [],
            }

        self.best_trial: int = self.tried_params_history["best_trial"]
        self.best_score: float = -np.inf

    def load_params_dict(self):
        with open(self.tried_params_file, "rb") as jsn:
            return json.load(jsn)

    def save_params_dict(self):
        with open(self.tried_params_file, "w") as jsn:
            json.dump(self.tried_params_history, jsn)

    def suggest_param(self, trial: BaseTrial, param_dict: dict):
        if param_dict["type"] == "float":
            suggested_param = trial.suggest_float(
                name=param_dict["name"],
                low=param_dict["low"],
                high=param_dict["high"],
            )
        elif param_dict["type"] == "categorical":
            suggested_param = trial.suggest_categorical(
                name=param_dict["name"],
                choices=param_dict["choices"],
            )

        elif param_dict["type"] == "int":
            suggested_param = trial.suggest_int(
                name=param_dict["name"],
                low=param_dict["low"],
                high=param_dict["high"],
            )

        return suggested_param

    def sample_params(self, trial: BaseTrial):
        suggested_params = {}
        for param_dict in self.param_dicts:
            suggested_params[param_dict["name"]] = self.suggest_param(trial, param_dict)

        return suggested_params

    def objective(self, trial: BaseTrial):
        print(f"Trial: {trial.number} has been started.")
        current_trial_dict = {}
        pruned: bool = False
        current_trial_dict["number"] = trial.number
        suggested_params = self.sample_params(trial)
        table = PrettyTable()
        field_names = ["Trial"] + list(suggested_params.keys()) + ["Time"]
        table.field_names = field_names
        row = [trial.number] + list(suggested_params.values())

        if suggested_params in self.tried_params_history["params_history"]:
            raise optuna.exceptions.TrialPruned("Pruned due to repeated parameters!!!")
        else:
            current_trial_dict["params"] = suggested_params
            self.tried_params_history["params_history"].append(suggested_params)

        env = gym.make(self.env_name)
        model: BaseAlgorithm = self.model_class(
            env,
            **suggested_params,
        )

        try:
            best_avg_eval_score = model.train(trial)

            if trial.should_prune():
                pruned = True
                current_trial_dict["pruned"] = pruned
                raise optuna.exceptions.TrialPruned("Pruned due to bad performance!!!")

            if best_avg_eval_score > self.best_score:
                self.best_score = best_avg_eval_score
                self.best_trial = trial.number

            self.tried_params_history["best_trial"] = self.best_trial
            current_trial_dict["score"] = best_avg_eval_score
            current_trial_dict["time_elapsed"] = model.time_elapsed

        except optuna.exceptions.TrialPruned:
            current_trial_dict["score"] = None
            current_trial_dict["time_elapsed"] = None

        finally:
            row = row + [model.time_elapsed]
            table.add_row(row)
            # print(table)

            self.tried_params_history["trials"].append(current_trial_dict)
            self.save_params_dict()

        return best_avg_eval_score

    def tune(self):
        sampler = optuna.samplers.TPESampler(self.sampler_seed)
        study = optuna.create_study(
            study_name=f"{self.model_class.algo_name}_{self.env_name}",
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(),
            storage=self.storage,
            load_if_exists=True,
        )
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            catch=(ValueError,),
        )
        print(f"Best Trial No: {study.best_trial.number}")
        print(f"Best Parameters: {study.best_params}")
        print(f"Best Score: {study.best_value}")
