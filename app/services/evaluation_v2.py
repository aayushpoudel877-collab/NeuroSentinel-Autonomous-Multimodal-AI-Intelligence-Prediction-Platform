from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error

@dataclass
class EvaluationReport:
    task: str
    metrics: dict[str, float]
    samples: int
    confusion_matrix: list[list[int]] | None = None
    def to_dict(self) -> dict[str, Any]: return asdict(self)

def evaluate_classification(y_true, y_pred) -> EvaluationReport:
    true=np.asarray(y_true); pred=np.asarray(y_pred)
    if len(true) != len(pred) or len(true) == 0: raise ValueError('prediction and target arrays must have equal non-zero length')
    cm=confusion_matrix(true,pred).tolist()
    metrics={'accuracy':float(accuracy_score(true,pred)),'precision_macro':float(precision_score(true,pred,average='macro',zero_division=0)),'recall_macro':float(recall_score(true,pred,average='macro',zero_division=0)),'f1_macro':float(f1_score(true,pred,average='macro',zero_division=0))}
    return EvaluationReport('classification',metrics,len(true),cm)

def evaluate_regression(y_true,y_pred) -> EvaluationReport:
    true=np.asarray(y_true,dtype=float); pred=np.asarray(y_pred,dtype=float)
    if true.shape != pred.shape or true.size == 0: raise ValueError('prediction and target arrays must have equal non-zero shape')
    metrics={'mae':float(mean_absolute_error(true,pred)),'rmse':float(np.sqrt(mean_squared_error(true,pred)))}
    return EvaluationReport('regression',metrics,int(true.size))
