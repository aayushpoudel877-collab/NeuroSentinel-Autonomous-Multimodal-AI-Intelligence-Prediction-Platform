from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    gradient_clip: float = 1.0
    device: str = 'auto'
    patience: int = 3

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1: raise ValueError('epochs and batch_size must be positive')
        if self.learning_rate <= 0 or self.weight_decay < 0: raise ValueError('invalid optimizer settings')
        if self.gradient_clip <= 0 or self.patience < 1: raise ValueError('gradient_clip and patience must be positive')
