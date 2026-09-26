from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class LearnedFusion:
    """Trainable late-fusion classifier over modality scores and confidences."""

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ])
        self.fitted = False

    def fit(self, features: list[list[float]], labels: list[str]) -> "LearnedFusion":
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or len(matrix) != len(labels):
            raise ValueError("features and labels must align")
        self.pipeline.fit(matrix, labels)
        self.fitted = True
        return self

    def predict(self, features: list[list[float]]) -> list[str]:
        if not self.fitted:
            raise RuntimeError("fusion model is not fitted")
        return self.pipeline.predict(np.asarray(features, dtype=float)).tolist()

    def probabilities(self, features: list[list[float]]) -> list[list[float]]:
        if not self.fitted:
            raise RuntimeError("fusion model is not fitted")
        return self.pipeline.predict_proba(np.asarray(features, dtype=float)).tolist()
