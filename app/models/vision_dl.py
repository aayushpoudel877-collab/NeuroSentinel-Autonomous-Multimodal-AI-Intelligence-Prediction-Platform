from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ImageFeatureClassifier:
    """Trainable vision baseline using compact image statistics."""

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ])
        self.fitted = False

    @staticmethod
    def features(images: list[np.ndarray]) -> np.ndarray:
        rows = []
        for image in images:
            arr = np.asarray(image, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.ndim != 3:
                raise ValueError("each image must have 2 or 3 dimensions")
            rows.append(np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1)), [arr.mean()]]))
        return np.asarray(rows, dtype=np.float32)

    def fit(self, images: list[np.ndarray], labels: list[str]) -> "ImageFeatureClassifier":
        self.pipeline.fit(self.features(images), labels)
        self.fitted = True
        return self

    def predict(self, images: list[np.ndarray]) -> list[str]:
        if not self.fitted:
            raise RuntimeError("vision classifier is not fitted")
        return self.pipeline.predict(self.features(images)).tolist()
