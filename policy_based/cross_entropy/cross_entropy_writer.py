import numpy as np

from utils.base_classes import BaseWriter
from utils.mlflow_logger import MLFlowLogger


class CrossEntropyWriter(BaseWriter):
    def __init__(
        self, writing_period: int, time_steps: int, mlflow_logger: MLFlowLogger
    ) -> None:
        super().__init__(
            writing_period=writing_period,
            time_steps=time_steps,
            mlflow_logger=mlflow_logger,
        )

        self.table.field_names = [
            "Time Step",
            "Score Bound",
            "Average Loss",
            "Average Train Score",
            "Average Evaluation Score",
            "Time Elapsed (s)",
        ]

        self.losses: list[float] = []

        self.avg_loss: float = 0.0
        self.score_bound: float = 0.0

    def reset(self, time_step: int):
        super().reset(time_step)

    def calculate_averages(self):
        super().calculate_averages()
        if len(self.losses) > 0:
            self.avg_loss = np.mean(self.losses[-100:])
        else:
            self.avg_loss = np.nan

        self.mlflow_loger.log_metric(
            "Average Loss",
            self.avg_loss,
            step=self.time_step,
        )

    def add_row_to_table(self):
        self.table.add_row(
            [
                f"{self.time_step}/{self.time_steps}",
                f"{self.score_bound:.4f}",
                f"{self.avg_loss:.4f}",
                f"{self.avg_train_score:.2f}",
                f"{self.avg_eval_score:.2f}",
                f"{self.time_elapsed:.2f}",
            ]
        )
