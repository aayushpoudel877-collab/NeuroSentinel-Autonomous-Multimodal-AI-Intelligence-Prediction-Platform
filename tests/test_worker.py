from datetime import datetime,timezone,timedelta
import pytest
from app.training.scheduler import TrainingScheduler
from app.training.worker import TrainingWorker

def test_worker_executes_and_completes_schedule(tmp_path):
    scheduler=TrainingScheduler(tmp_path/"schedules.json"); schedule=scheduler.create("demo",1)
    scheduler.get(schedule.schedule_id).next_run_at=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
    result=TrainingWorker(scheduler).run_one(schedule.schedule_id,lambda item:{"job":{"job_id":"job-123"},"model":item["model"]})
    assert result["schedule"]["last_job_id"]=="job-123"
    assert result["schedule"]["lease_id"] is None

def test_worker_releases_lease_on_failure(tmp_path):
    scheduler=TrainingScheduler(tmp_path/"schedules.json"); schedule=scheduler.create("demo",1)
    scheduler.get(schedule.schedule_id).next_run_at=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
    with pytest.raises(RuntimeError):
        TrainingWorker(scheduler).run_one(schedule.schedule_id,lambda _:(_ for _ in ()).throw(RuntimeError("boom")))
    assert scheduler.get(schedule.schedule_id).lease_id is None
    assert scheduler.get(schedule.schedule_id).failure_count==1
