from __future__ import annotations
from typing import Any,Callable
from app.training.scheduler import TrainingScheduler

class TrainingWorker:
    """Worker facade for safely claiming and executing due schedules."""

    def __init__(self,scheduler:TrainingScheduler)->None: self.scheduler=scheduler
    def due(self)->list[dict[str,Any]]: return [x for x in self.scheduler.list() if self.scheduler.due(x["schedule_id"])]
    def run_one(self,schedule_id:str,trainer:Callable[[dict[str,Any]],dict[str,Any]])->dict[str,Any]:
        lease_id=self.scheduler.claim(schedule_id); schedule=self.scheduler.get(schedule_id).to_dict()
        try:
            result=trainer(schedule)
            job_id=str(result.get("job",{}).get("job_id") or result.get("job_id") or "")
            if not job_id: raise ValueError("training result must contain a job_id")
            updated=self.scheduler.complete(schedule_id,lease_id,job_id)
            return {"schedule":updated.to_dict(),"result":result}
        except Exception:
            self.scheduler.fail(schedule_id,lease_id); raise
