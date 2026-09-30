from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import uuid
from typing import Any,Iterator

@dataclass
class TrainingSchedule:
    schedule_id:str
    model:str
    interval_minutes:int
    enabled:bool
    created_at:str
    next_run_at:str
    last_run_at:str|None=None
    last_job_id:str|None=None
    lease_id:str|None=None
    lease_expires_at:str|None=None
    failure_count:int=0
    def to_dict(self)->dict[str,Any]: return asdict(self)

class TrainingScheduler:
    def __init__(self,path:str|Path="artifacts/training/schedules.json",lease_minutes:int=15)->None:
        if lease_minutes<1: raise ValueError("lease_minutes must be positive")
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock_path=self.path.with_suffix(self.path.suffix+".lock")
        self.lease_minutes=lease_minutes; self._schedules={}; self._load()

    def _load(self)->None:
        if self.path.exists():
            payload=json.loads(self.path.read_text(encoding="utf-8"))
            self._schedules={x["schedule_id"]:TrainingSchedule(**x) for x in payload.get("schedules",[])}

    def _save(self)->None:
        tmp=self.path.with_suffix(self.path.suffix+".tmp")
        tmp.write_text(json.dumps({"schedules":[x.to_dict() for x in self._schedules.values()]},indent=2,sort_keys=True)+"\n",encoding="utf-8")
        tmp.replace(self.path)

    @contextmanager
    def _file_lock(self)->Iterator[None]:
        self.lock_path.touch(exist_ok=True)
        handle=self.lock_path.open("r+")
        try:
            try:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
            except ImportError:
                import msvcrt
                handle.seek(0)
                if handle.read(1)=="":
                    handle.seek(0); handle.write("0"); handle.flush()
                handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_LOCK,1)
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
            except ImportError:
                import msvcrt
                handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            handle.close()

    def create(self,model:str,interval_minutes:int,enabled:bool=True)->TrainingSchedule:
        if not model.strip(): raise ValueError("model is required")
        if interval_minutes<1: raise ValueError("interval_minutes must be positive")
        now=datetime.now(timezone.utc)
        schedule=TrainingSchedule(f"schedule-{uuid.uuid4().hex[:12]}",model,interval_minutes,enabled,now.isoformat(),(now+timedelta(minutes=interval_minutes)).isoformat())
        with self._file_lock():
            self._load(); self._schedules[schedule.schedule_id]=schedule; self._save()
        return schedule

    def list(self)->list[dict[str,Any]]:
        with self._file_lock():
            self._load(); return [x.to_dict() for x in self._schedules.values()]

    def get(self,schedule_id:str)->TrainingSchedule:
        if schedule_id not in self._schedules: raise KeyError(f"schedule not found: {schedule_id}")
        return self._schedules[schedule_id]

    def due(self,schedule_id:str,now:datetime|None=None)->bool:
        schedule=self.get(schedule_id)
        if not schedule.enabled: return False
        current=now or datetime.now(timezone.utc)
        if schedule.lease_expires_at and datetime.fromisoformat(schedule.lease_expires_at)>current: return False
        return current>=datetime.fromisoformat(schedule.next_run_at)

    def claim(self,schedule_id:str,now:datetime|None=None)->str:
        current=now or datetime.now(timezone.utc)
        with self._file_lock():
            self._load(); schedule=self.get(schedule_id)
            if not self.due(schedule_id,current): raise ValueError("schedule is not due or is already leased")
            lease_id=f"lease-{uuid.uuid4().hex[:12]}"
            schedule.lease_id=lease_id; schedule.lease_expires_at=(current+timedelta(minutes=self.lease_minutes)).isoformat()
            self._save(); return lease_id

    def complete(self,schedule_id:str,lease_id:str,job_id:str,when:datetime|None=None)->TrainingSchedule:
        if not job_id.strip(): raise ValueError("job_id is required")
        with self._file_lock():
            self._load(); schedule=self.get(schedule_id)
            if schedule.lease_id!=lease_id: raise ValueError("invalid or expired schedule lease")
            current=when or datetime.now(timezone.utc)
            schedule.last_run_at=current.isoformat(); schedule.last_job_id=job_id; schedule.next_run_at=(current+timedelta(minutes=schedule.interval_minutes)).isoformat()
            schedule.lease_id=None; schedule.lease_expires_at=None; schedule.failure_count=0; self._save(); return schedule

    def fail(self,schedule_id:str,lease_id:str)->TrainingSchedule:
        with self._file_lock():
            self._load(); schedule=self.get(schedule_id)
            if schedule.lease_id!=lease_id: raise ValueError("invalid or expired schedule lease")
            schedule.lease_id=None; schedule.lease_expires_at=None; schedule.failure_count+=1; self._save(); return schedule

    def mark_run(self,schedule_id:str,when:datetime|None=None)->TrainingSchedule:
        with self._file_lock():
            self._load(); schedule=self.get(schedule_id); current=when or datetime.now(timezone.utc)
            schedule.last_run_at=current.isoformat(); schedule.next_run_at=(current+timedelta(minutes=schedule.interval_minutes)).isoformat(); schedule.lease_id=None; schedule.lease_expires_at=None; self._save(); return schedule

    def set_enabled(self,schedule_id:str,enabled:bool)->TrainingSchedule:
        with self._file_lock():
            self._load(); schedule=self.get(schedule_id); schedule.enabled=enabled
            if not enabled: schedule.lease_id=None; schedule.lease_expires_at=None
            self._save(); return schedule
