import pytest
from app.mlops.registry import ModelRecord,ModelRegistry
from app.training.retraining import RetrainingManager

def test_retraining_manager_lifecycle(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json")
    registry.register(ModelRecord("demo","1","classification"))
    manager=RetrainingManager(registry,tmp_path/"jobs")
    job=manager.plan("demo",{"retrain":True,"reasons":["drift_threshold_exceeded"]})
    assert job.status=="queued"
    assert manager.start(job.job_id).status=="running"
    candidate=manager.mark_candidate(job.job_id,"2",artifact_uri="models/demo-2.bin",metrics={"f1":0.91},run_id="run-1")
    assert candidate.status=="candidate_ready"
    manager.register_candidate(job.job_id)
    assert registry.get("demo","2").parent_version=="1"
    assert registry.get("demo","2").status=="candidate"

def test_not_required_job_does_not_start(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json")
    registry.register(ModelRecord("demo","1","classification"))
    manager=RetrainingManager(registry,tmp_path/"jobs")
    job=manager.plan("demo",{"retrain":False,"reasons":[]})
    assert job.status=="not_required"
    with pytest.raises(ValueError):
        manager.start(job.job_id)
