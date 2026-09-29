import numpy as np
import pytest
from app.mlops.registry import ModelRecord,ModelRegistry
from app.training.retraining import RetrainingManager
from app.training.control_plane import TrainingControlPlane

def test_control_plane_runs_and_evaluates_forecast(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json")
    registry.register(ModelRecord("forecast","1.0","time-series",status="production"))
    manager=RetrainingManager(registry,tmp_path/"jobs")
    control=TrainingControlPlane(registry,manager,artifact_root=tmp_path/"models",experiment_root=tmp_path/"experiments",schedule_path=tmp_path/"schedules.json")
    values=(np.sin(np.arange(60)/3.0)*10+50).tolist()
    result=control.run_forecast("forecast",values)
    version=result["candidate"]["version"]
    assert result["job"]["status"]=="candidate_ready"
    assert result["candidate"]["artifact_sha256"]
    gate=control.evaluate_candidate("forecast",version,metric="mae",threshold=100.0,greater_is_better=False)
    assert gate["passed"] is True
    assert gate["action"]=="staged"
    assert registry.get("forecast",version).status=="staging"

def test_control_plane_rejects_wrong_task(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json")
    registry.register(ModelRecord("text","1.0","NLP",status="production"))
    manager=RetrainingManager(registry,tmp_path/"jobs")
    control=TrainingControlPlane(registry,manager,schedule_path=tmp_path/"schedules.json")
    with pytest.raises(ValueError,match="time-series"):
        control.run_forecast("text",[1.0]*20)

def test_control_plane_cannot_stage_production_version(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json")
    registry.register(ModelRecord("forecast","1.0","time-series",status="production",metrics={"mae":1.0}))
    manager=RetrainingManager(registry,tmp_path/"jobs")
    control=TrainingControlPlane(registry,manager,schedule_path=tmp_path/"schedules.json")
    with pytest.raises(ValueError,match="only candidate models"):
        control.evaluate_candidate("forecast","1.0",metric="mae",threshold=2.0,greater_is_better=False)
    assert registry.get("forecast","1.0").status=="production"
