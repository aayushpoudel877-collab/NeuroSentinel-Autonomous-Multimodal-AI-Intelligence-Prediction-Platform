from __future__ import annotations

import numpy as np

from app.models.temporal import TemporalRegressor


def synthetic_series(n: int = 120, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 0.03 * t + np.sin(t / 5.0) + rng.normal(0, 0.08, n)
    return values.tolist()


def train_demo() -> TemporalRegressor:
    model = TemporalRegressor(lags=12).fit(synthetic_series())
    return model


if __name__ == "__main__":
    model = train_demo()
    print(model.forecast(synthetic_series(), horizon=5))
