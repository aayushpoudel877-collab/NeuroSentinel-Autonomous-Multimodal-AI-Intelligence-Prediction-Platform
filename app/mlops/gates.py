from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from app.mlops.registry import ModelRecord

@dataclass(frozen=True)
class PromotionGate:
    metric: str
    minimum: float
    greater_is_better: bool = True
    def evaluate(self, metrics: Mapping[str,float]) -> tuple[bool,str]:
        if self.metric not in metrics: return False, f"required metric missing: {self.metric}"
        value=float(metrics[self.metric]); passed=value>=self.minimum if self.greater_is_better else value<=self.minimum
        comparison=">=" if self.greater_is_better else "<="
        return passed, f"{self.metric}={value:.6g} {comparison} {self.minimum:.6g}"

class PromotionPolicy:
    def __init__(self, gates: Mapping[str,PromotionGate] | None = None) -> None: self.gates=dict(gates or {})
    def check(self, record: ModelRecord) -> dict[str,object]:
        gate=self.gates.get(record.task)
        if gate is None: return {"passed":True,"reason":"no task-specific gate configured"}
        passed,reason=gate.evaluate(record.metrics or {})
        return {"passed":passed,"reason":reason,"metric":gate.metric}
    def require(self, record: ModelRecord) -> None:
        result=self.check(record)
        if not result["passed"]: raise ValueError(f"promotion gate failed: {result['reason']}")
