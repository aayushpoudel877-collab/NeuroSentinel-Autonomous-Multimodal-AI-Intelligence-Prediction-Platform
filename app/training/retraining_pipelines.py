from __future__ import annotations
from pathlib import Path
from typing import Any
import uuid
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from app.experiments.tracker import ExperimentTracker
from app.artifacts.store import ArtifactStore

class ForecastRetrainingPipeline:
    """Concrete lightweight retraining pipeline for the lagged Ridge forecaster."""

    def __init__(self, artifact_root: str|Path="models/artifacts", experiment_root: str|Path="artifacts/experiments"):
        self.artifacts=ArtifactStore(artifact_root)
        self.experiments=ExperimentTracker(experiment_root)

    def run(self, job: Any, values: list[float], window: int=6) -> dict[str,Any]:
        series=np.asarray(values,dtype=float)
        if series.ndim!=1 or series.size<=window+1:
            raise ValueError("forecast retraining requires more than window+1 observations")
        X=np.array([series[i-window:i] for i in range(window,len(series))])
        y=series[window:]
        split=max(int(len(X)*0.8),1)
        if split>=len(X): split=len(X)-1
        train_x,valid_x=X[:split],X[split:]
        train_y,valid_y=y[:split],y[split:]
        model=Ridge(alpha=1.0).fit(train_x,train_y)
        predictions=model.predict(valid_x)
        mae=float(np.mean(np.abs(valid_y-predictions)))
        rmse=float(np.sqrt(np.mean((valid_y-predictions)**2)))
        run_id=f"run-{uuid.uuid4().hex[:12]}"
        version=f"{job.source_version}-r{uuid.uuid4().hex[:8]}"
        artifact=self.artifacts.root/f"{job.model}-{version}.joblib"
        joblib.dump(model,artifact)
        metrics={"mae":mae,"rmse":rmse}
        self.experiments.log(run_id,job.model,{"window":window,"alpha":1.0,"samples":int(series.size)},metrics)
        return {"version":version,"artifact_uri":str(artifact),"metrics":metrics,"run_id":run_id}
