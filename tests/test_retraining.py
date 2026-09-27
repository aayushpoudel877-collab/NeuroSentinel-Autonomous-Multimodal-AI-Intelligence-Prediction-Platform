from app.mlops.registry import ModelRecord,ModelRegistry
from app.training.retraining import RetrainingManager

def test_retraining_manager_queues_job(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json")
    registry.register(ModelRecord("demo","1","classification"))
    manager=RetrainingManager(registry,tmp_path/"jobs")
    job=manager.plan("demo",{"retrain":True,"reasons":["drift_threshold_exceeded"]})
    assert job.status=="queued"
    assert manager.load(job.job_id).model=="demo"
