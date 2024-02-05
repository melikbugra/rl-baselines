from datetime import datetime
from typing import Any

from gymnasium import Env
import mlflow


class MLFlowLogger:
    def __init__(self, mlflow_tracking_uri: str) -> None:
        self.run_id: str
        self.log: bool = False
        if mlflow_tracking_uri:
            self.log = True
            mlflow.set_tracking_uri(mlflow_tracking_uri)

    def define_experiment_and_run(
        self, params_to_log: dict[str, Any], env: Env, algo_name: str
    ):
        experiment_name = (
            f"{env.unwrapped.spec.id}_{algo_name.lower().replace(' ', '_')}"
        )
        mlflow.set_experiment(experiment_name)
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.active_run = mlflow.start_run(run_name=run_name)
        for param_name, param in params_to_log.items():
            mlflow.log_param(param_name, param)
        self.run_id = self.active_run.info.run_id

    def log_metric(self, key: str, value: float, step: int | None = None):
        if self.log:
            mlflow.log_metric(key=key, value=value, step=step, run_id=self.run_id)

    def log_param(self, param_name: str, param: Any):
        if self.log:
            mlflow.log_param(param_name, param)

    def log_params(self, params_to_log: dict[str, Any]):
        if self.log:
            for param_name, param in params_to_log.items():
                mlflow.log_param(param_name, param)

    def end_run(self):
        if self.log:
            mlflow.end_run()
