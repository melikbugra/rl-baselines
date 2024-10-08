from abc import ABC, abstractmethod

import numpy as np
from prettytable import PrettyTable

from utils.mlflow_logger.mlflow_logger import MLFlowLogger


class BaseWriter(ABC):
    def __init__(
        self, writing_period: int, time_steps: int, mlflow_logger: MLFlowLogger
    ) -> None:
        super().__init__()
        self.table = PrettyTable()

        if writing_period >= time_steps:
            self.time_step: int = time_steps
        else:
            self.time_step: int = writing_period
        self.time_steps: int = time_steps

        self.train_scores: list[float] = []
        self.avg_eval_score: float = 0.0

        self.avg_train_score: float = 0.0

        self.mlflow_loger: MLFlowLogger = mlflow_logger

        self.time_elapsed: float

    def __str__(self) -> str:
        self.calculate_averages()
        self.add_row_to_table()
        return f"{str(self.table)}\n"

    @abstractmethod
    def reset(self, time_step: int):
        self.time_step = time_step
        self.table.clear_rows()

    @abstractmethod
    def calculate_averages(self):
        if len(self.train_scores) > 0:
            self.avg_train_score = np.mean(self.train_scores[-20:])
        else:
            self.avg_train_score = np.nan

    @abstractmethod
    def add_row_to_table(self):
        pass
