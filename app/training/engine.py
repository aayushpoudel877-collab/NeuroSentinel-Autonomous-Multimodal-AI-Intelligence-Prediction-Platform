from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class TrainingState:
    epoch: int = 0
    best_metric: float | None = None
    stale_epochs: int = 0

class TrainingEngine:
    def __init__(self, config: Any, mode: str = "max") -> None:
        config.validate()
        if mode not in {"max", "min"}:
            raise ValueError("mode must be max or min")
        self.config = config
        self.mode = mode
        self.state = TrainingState()

    def should_stop(self) -> bool:
        return self.state.stale_epochs >= self.config.patience

    def update(self, metric: float) -> bool:
        if self.state.best_metric is None:
            improved = True
        elif self.mode == "max":
            improved = metric > self.state.best_metric
        else:
            improved = metric < self.state.best_metric
        if improved:
            self.state.best_metric = float(metric)
            self.state.stale_epochs = 0
        else:
            self.state.stale_epochs += 1
        self.state.epoch += 1
        return improved
