from typing import Type

from gymnasium import Env
import optuna
from optuna.trial import BaseTrial

from utils.base_classes.base_algorithm import BaseAlgorithm


class Tuner:
    def __init__(
        self,
        env: Env,
        model_class: Type[BaseAlgorithm],
        param_dicts: list[dict],
        sampler_seed: int = 42,
        n_trials: int = 100,
        n_jobs: int = 1,
        mlflow_tracking_uri: str = None,
    ) -> None:
        self.tried_params: list = []
        self.param_dicts = param_dicts
        self.env: Env = env
        self.model_class: Type[BaseAlgorithm] = model_class
        self.sampler_seed: int = sampler_seed
        self.n_trials: int = n_trials
        self.n_jobs: int = n_jobs
        self.mlflow_tracking_uri: str = mlflow_tracking_uri

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
        suggested_params = self.sample_params(trial)

        if suggested_params in self.tried_params:
            raise optuna.exceptions.TrialPruned("Pruned due to repeated parameters!!!")
        else:
            self.tried_params.append(suggested_params)

        model: BaseAlgorithm = self.model_class(
            self.env,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            **suggested_params,
        )
        writing_period = int(model.time_steps / 10)
        model.writing_period = writing_period
        model.writer.time_step = writing_period
        best_avg_eval_score = model.train(trial)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned("Pruned due to bad performance!!!")

        return best_avg_eval_score

    def tune(self):
        sampler = optuna.samplers.TPESampler(self.sampler_seed)
        study = optuna.create_study(
            study_name=self.env.spec.id,
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(),
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
