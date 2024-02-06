from datetime import datetime
from typing import Type, Any

from gymnasium import Env
import optuna
from optuna.trial import BaseTrial
import mlflow
import numpy as np

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

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.define_experiment(
            algo_name=model_class.algo_name,
        )

        self.best_trial: int = 0
        self.best_trial_score: float = -np.inf

    def define_experiment(self, algo_name: str):
        experiment_name = f"Tuning: {self.env.unwrapped.spec.id}_{algo_name.lower().replace(' ', '_')}"
        mlflow.set_experiment(experiment_name)

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
        pruned: bool = False
        suggested_params = self.sample_params(trial)
        print(f"Trial: {trial.number}:")
        for param_name, param in suggested_params.items():
            print(f"\t{param_name}: {param}")

        if suggested_params in self.tried_params:
            raise optuna.exceptions.TrialPruned("Pruned due to repeated parameters!!!")
        else:
            self.tried_params.append(suggested_params)

        model: BaseAlgorithm = self.model_class(
            self.env,
            **suggested_params,
        )

        run_name = f"Trial: {trial.number}"
        with mlflow.start_run(run_name=run_name):
            for param_name, param in suggested_params.items():
                mlflow.log_param(param_name, param)

            model.writing_period = model.time_steps
            model.writer.time_step = model.time_steps
            best_avg_eval_score = model.train(trial)

            if trial.should_prune():
                pruned = True
                mlflow.log_param("Best Average Evaluation Score", None)
                mlflow.log_param("Pruned", pruned)
                raise optuna.exceptions.TrialPruned("Pruned due to bad performance!!!")
            else:
                mlflow.log_param("Best Average Evaluation Score", best_avg_eval_score)
                mlflow.log_param("Pruned", pruned)

            if best_avg_eval_score > self.best_trial_score:
                self.best_trial_score = best_avg_eval_score
                self.best_trial = trial.number

            mlflow.log_param("Best Trial", self.best_trial)

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
