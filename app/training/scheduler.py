from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import uuid
from typing import Any

@dataclass
class TrainingSchedule:
    schedule_id:str
    model:str
    interval_minutes:int
    enabled:bool
    created_at:str
    next_run_at:str
    last_run_at:str|None=None
    def to_dict(self)->dict[str,Any]: return asdict(self)

class TrainingScheduler:
    def __init__(self,path:str|Path="artifacts/training/schedules.json")->None:
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._schedules={}; self._load()

    def _load(self)->None:
        if self.path.exists():
            payload=json.loads(self.path.read_text(encoding="utf-8"))
            self._schedules={x["schedule_id"]:TrainingSchedule(**x) for x in payload.get("schedules",[])}

    def _save(self)->None:
        tmp=self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"schedules":[x.to_dict() for x in self._schedules.values()]},indent=2,sort_keys=True)+"\n",encoding="utf-8")
        tmp.replace(self.path)

    def create(self,model:str,interval_minutes:int,enabled:bool=True)->TrainingSchedule:
        if interval_minutes<1: raise ValueError("interval_minutes must be positive")
        now=datetime.now(timezone.utc)
        schedule=TrainingSchedule(
            schedule_id=f"schedule-{uuid.uuid4().hex[:12]}",
            model=model,interval_minutes=interval_minutes,enabled=enabled,
            created_at=now.isoformat(),next_run_at=(now+timedelta(minutes=interval_minutes)).isoformat()
        )
        self._schedules[schedule.schedule_id]=schedule; self._save(); return schedule

    def list(self)->list[dict[str,Any]]:
        return [x.to_dict() for x in self._schedules.values()]

    def get(self,schedule_id:str)->TrainingSchedule:
        if schedule_id not in self._schedules: raise KeyError(f"schedule not found: {schedule_id}")
        return self._schedules[schedule_id]

    def due(self,schedule_id:str,now:datetime|None=None)->bool:
        schedule=self.get(schedule_id)
        if not schedule.enabled: return False
        current=now or datetime.now(timezone.utc)
        due_at=datetime.fromisoformat(schedule.next_run_at)
        return current>=due_at

    def mark_run(self,schedule_id:str,when:datetime|None=None)->TrainingSchedule:
        schedule=self.get(schedule_id)
        current=when or datetime.now(timezone.utc)
        schedule.last_run_at=current.isoformat()
        schedule.next_run_at=(current+timedelta(minutes=schedule.interval_minutes)).isoformat()
        self._save(); return schedule

    def set_enabled(self,schedule_id:str,enabled:bool)->TrainingSchedule:
        schedule=self.get(schedule_id); schedule.enabled=enabled; self._save(); return schedule
