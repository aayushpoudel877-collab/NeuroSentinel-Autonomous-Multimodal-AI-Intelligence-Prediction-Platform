import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def detect(self, values):
        x = np.asarray(values, dtype=float).reshape(-1, 1)
        model = IsolationForest(contamination="auto", random_state=42, n_estimators=150)
        labels = model.fit_predict(x)
        scores = -model.score_samples(x)
        return {
            "task": "anomaly_detection",
            "anomalies": [i for i, label in enumerate(labels) if label == -1],
            "scores": [round(float(s), 5) for s in scores]
        }
