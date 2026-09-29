from datetime import datetime,timezone,timedelta
import pytest
from app.training.scheduler import TrainingScheduler

def test_schedule_lifecycle(tmp_path):
    scheduler=TrainingScheduler(tmp_path/"schedules.json"); schedule=scheduler.create("demo",30)
    assert not scheduler.due(schedule.schedule_id)
    due_time=datetime.now(timezone.utc)+timedelta(minutes=31)
    assert scheduler.due(schedule.schedule_id,due_time)
    lease=scheduler.claim(schedule.schedule_id,due_time)
    assert not scheduler.due(schedule.schedule_id,due_time)
    updated=scheduler.complete(schedule.schedule_id,lease,"job-1",due_time)
    assert updated.last_job_id=="job-1"
    assert not scheduler.due(schedule.schedule_id,due_time)

def test_disabled_schedule_not_due(tmp_path):
    scheduler=TrainingScheduler(tmp_path/"schedules.json"); schedule=scheduler.create("demo",1,enabled=False)
    assert not scheduler.due(schedule.schedule_id,datetime.now(timezone.utc)+timedelta(days=1))

def test_failed_lease_can_retry(tmp_path):
    scheduler=TrainingScheduler(tmp_path/"schedules.json"); schedule=scheduler.create("demo",1)
    now=datetime.now(timezone.utc)+timedelta(minutes=2); lease=scheduler.claim(schedule.schedule_id,now); scheduler.fail(schedule.schedule_id,lease)
    assert scheduler.due(schedule.schedule_id,now); assert scheduler.get(schedule.schedule_id).failure_count==1

def test_double_claim_is_rejected(tmp_path):
    scheduler=TrainingScheduler(tmp_path/"schedules.json"); schedule=scheduler.create("demo",1); now=datetime.now(timezone.utc)+timedelta(minutes=2)
    scheduler.claim(schedule.schedule_id,now)
    with pytest.raises(ValueError,match="not due"): scheduler.claim(schedule.schedule_id,now)
