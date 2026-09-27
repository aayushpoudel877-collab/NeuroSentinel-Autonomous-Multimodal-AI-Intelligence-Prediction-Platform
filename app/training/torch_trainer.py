from __future__ import annotations

from typing import Any
from app.training.engine import TrainingEngine
from app.training.device import resolve_device

class TorchTrainer:
    def __init__(self, model: Any, optimizer: Any, criterion: Any, engine: TrainingEngine) -> None:
        self.model=model; self.optimizer=optimizer; self.criterion=criterion; self.engine=engine
        self.device=resolve_device(engine.config.device)
        self.model.to(self.device)

    def train_epoch(self, loader: Any) -> float:
        self.model.train(); total=0.0; count=0
        import torch
        for batch in loader:
            inputs, targets=batch
            inputs, targets=inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outputs=self.model(inputs)
            loss=self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.engine.config.gradient_clip)
            self.optimizer.step()
            total += float(loss.detach().cpu()) * targets.shape[0]; count += targets.shape[0]
        return total / max(count, 1)

    def evaluate(self, loader: Any) -> float:
        self.model.eval(); total=0.0; count=0
        with __import__('torch').no_grad():
            for batch in loader:
                inputs, targets=batch
                inputs, targets=inputs.to(self.device), targets.to(self.device)
                loss=self.criterion(self.model(inputs), targets)
                total += float(loss.detach().cpu()) * targets.shape[0]; count += targets.shape[0]
        return total / max(count, 1)
