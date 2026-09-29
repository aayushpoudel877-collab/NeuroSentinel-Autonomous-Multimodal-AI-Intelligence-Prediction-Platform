from datetime import datetime,timezone,timedelta
from app.training.scheduler import TrainingScheduler

def test_schedule_lifecycle(tmp_path):
    scheduler=TrainingScheduler(tmp_path/"schedules.json")
    schedule=scheduler.create("demo",30)
    assert schedule.interval_minutes==30
    assert not scheduler.due(schedule.schedule_id)
    due_time=datetime.now(timezone.utc)+timedelta(minutes=31)
    assert scheduler.due(schedule.schedule_id,due_time)
    updated=scheduler.mark_run(schedule.schedule_id,due_time)
    assert updated.last_run_at==due_time.isoformat()
    assert not scheduler.due(schedule.schedule_id,due_time)

def test_disabled_schedule_not_due(tmp_path):
    scheduler=TrainingScheduler(tmp_path/"schedules.json")
    schedule=scheduler.create("demo",1,enabled=False)
    assert not scheduler.due(schedule.schedule_id,datetime.now(timezone.utc)+timedelta(days=1))
