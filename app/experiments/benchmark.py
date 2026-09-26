from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class BenchmarkResult:
    dataset: str
    model: str
    accuracy: float
    macro_f1: float
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_tabular_benchmark(seed: int = 42) -> BenchmarkResult:
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=seed, stratify=data.target
    )
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    model.fit(x_train, y_train)
    start = perf_counter()
    predictions = model.predict(x_test)
    latency_ms = (perf_counter() - start) * 1000 / max(len(x_test), 1)
    return BenchmarkResult(
        dataset="sklearn:breast_cancer",
        model="standardized-logistic-regression",
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro")),
        latency_ms=float(latency_ms),
    )


if __name__ == "__main__":
    print(run_tabular_benchmark().to_dict())
