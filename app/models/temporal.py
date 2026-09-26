from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class TemporalRegressor:
    """Trainable lag-based temporal regressor with recursive forecasting."""

    def __init__(self, lags: int = 12, alpha: float = 1.0) -> None:
        if lags < 2:
            raise ValueError("lags must be >= 2")
        self.lags = lags
        self.model = Ridge(alpha=alpha)
        self.fitted = False

    def fit(self, values: list[float]) -> "TemporalRegressor":
        series = np.asarray(values, dtype=float)
        if len(series) <= self.lags:
            raise ValueError("values must contain more samples than lags")
        x = np.array([series[i - self.lags:i] for i in range(self.lags, len(series))])
        y = series[self.lags:]
        self.model.fit(x, y)
        self.fitted = True
        return self

    def forecast(self, values: list[float], horizon: int = 5) -> list[float]:
        if not self.fitted:
            raise RuntimeError("temporal regressor is not fitted")
        history = list(map(float, values))
        if len(history) < self.lags:
            raise ValueError("values must contain at least lags samples")
        predictions: list[float] = []
        for _ in range(horizon):
            pred = float(self.model.predict(np.asarray(history[-self.lags:]).reshape(1, -1))[0])
            predictions.append(pred)
            history.append(pred)
        return predictions
