from abc import ABC, abstractmethod

import numpy as np
from prettytable import PrettyTable


class BaseWriter(ABC):
    def __init__(self, writing_period: int, time_steps: int) -> None:
        super().__init__()
        self.table = PrettyTable()

        self.time_step: int = writing_period
        self.time_steps: int = time_steps

        self.train_scores: list[float] = []
        self.eval_score: float = []

        self.avg_train_score: float = 0.0

        self.mlflow_run_id: str = None

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
        self.avg_train_score = np.mean(self.train_scores[-20:])

    @abstractmethod
    def add_row_to_table(self):
        pass
