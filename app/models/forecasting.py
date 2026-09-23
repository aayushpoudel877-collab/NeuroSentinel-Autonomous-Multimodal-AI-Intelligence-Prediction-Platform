import numpy as np
from sklearn.linear_model import Ridge

class Forecaster:
    def predict(self, values, horizon=5):
        y = np.asarray(values, dtype=float)
        window = min(6, len(y) - 1)
        X, target = [], []
        for i in range(window, len(y)):
            X.append(y[i-window:i])
            target.append(y[i])
        model = Ridge(alpha=1.0).fit(X, target)
        history = list(y[-window:])
        predictions = []
        for _ in range(horizon):
            pred = float(model.predict([history[-window:]])[0])
            predictions.append(pred)
            history.append(pred)
        return {"task": "forecasting", "horizon": horizon, "predictions": predictions}
