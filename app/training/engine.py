from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class TrainingState:
    epoch: int = 0
    best_metric: float | None = None
    stale_epochs: int = 0

class TrainingEngine:
    def __init__(self, config: Any) -> None:
        config.validate()
        self.config = config
        self.state = TrainingState()

    def should_stop(self) -> bool:
        return self.state.stale_epochs >= self.config.patience

    def update(self, metric: float) -> bool:
        improved = self.state.best_metric is None or metric > self.state.best_metric
        if improved:
            self.state.best_metric = metric
            self.state.stale_epochs = 0
        else:
            self.state.stale_epochs += 1
        self.state.epoch += 1
        return improved
