import numpy as np
import mlflow

from utils.base_classes.base_writer import BaseWriter


class DQNWriter(BaseWriter):
    def __init__(self, writing_period: int, time_steps: int) -> None:
        super().__init__(writing_period=writing_period, time_steps=time_steps)

        self.table.field_names = [
            "Time Step",
            "Epsilon",
            "Average Loss",
            "Average Train Score",
            "Evaluation Score",
        ]

        self.losses: list[float] = []

        self.avg_loss: float = 0.0
        self.epsilon: float = 0.0

    def reset(self, time_step: int):
        super().reset(time_step)

    def calculate_averages(self):
        super().calculate_averages()
        self.avg_loss = np.mean(self.losses[-100:])
        mlflow.log_metric(
            "Average Loss",
            self.avg_loss,
            step=self.time_step,
            run_id=self.mlflow_run_id,
        )

    def add_row_to_table(self):
        self.table.add_row(
            [
                f"{self.time_step}/{self.time_steps}",
                f"{self.epsilon:.4f}",
                f"{self.avg_loss:.4f}",
                f"{self.avg_train_score:.2f}",
                f"{self.eval_score:.2f}",
            ]
        )
