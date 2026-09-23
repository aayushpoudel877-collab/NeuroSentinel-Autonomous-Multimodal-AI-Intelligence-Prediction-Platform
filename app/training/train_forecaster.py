from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge

def train(values, window=6, output="models/artifacts/forecaster.joblib"):
    y = np.asarray(values, dtype=float)
    X = np.array([y[i-window:i] for i in range(window, len(y))])
    target = y[window:]
    model = Ridge(alpha=1.0).fit(X, target)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return output

if __name__ == "__main__":
    train([10,11,12,13,14,15,16,18,17,20,21,22,23,25])
