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

        self.best_score: float = -np.inf

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
        suggested_params = self.sample_params(trial)

        for completed_trial in trial.study.get_trials(deepcopy=False):
            if completed_trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            if completed_trial.params == trial.params:
                print(f"Found duplicate trial with value: {completed_trial.value}")
                return completed_trial.value

        env = gym.make(self.env_name)
        model: BaseAlgorithm = self.model_class(
            env,
            **suggested_params,
        )

        best_avg_eval_score = model.train(trial)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned("Pruned due to bad performance!!!")

        if best_avg_eval_score > self.best_score:
            self.best_score = best_avg_eval_score
            self.best_trial = trial.number

        print(
            f"Trial {trial.number} has finished with score of {best_avg_eval_score} in {model.time_elapsed} seconds."
        )

        return -(best_avg_eval_score**2) / model.time_elapsed

    def tune(self):
        sampler = optuna.samplers.TPESampler(self.sampler_seed)
        study = optuna.create_study(
            study_name=f"{self.model_class.algo_name}_{self.env_name}",
            direction="minimize",
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
