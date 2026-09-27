from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
import uuid
from pathlib import Path
import json
from typing import Any
from app.mlops.registry import ModelRecord,ModelRegistry
from app.monitoring.policy import RetrainingDecision

@dataclass
class RetrainingJob:
    job_id:str
    model:str
    task:str
    reason:list[str]
    status:str
    created_at:str
    source_version:str|None=None
    candidate_version:str|None=None
    def to_dict(self)->dict[str,Any]: return asdict(self)

class RetrainingManager:
    def __init__(self,registry:ModelRegistry,root:str|Path="artifacts/retraining")->None:
        self.registry=registry; self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def plan(self,model:str,decision:dict[str,Any])->RetrainingJob:
        record=self.registry.get(model); now=datetime.now(timezone.utc).isoformat(); job_id=f"retrain-{uuid.uuid4().hex[:12]}"
        status="queued" if decision.get("retrain") else "not_required"
        job=RetrainingJob(job_id,model,record.task,decision.get("reasons",[]),status,now,record.version,None)
        self._write(job); return job
    def mark_candidate(self,job_id:str,version:str)->RetrainingJob:
        job=self.load(job_id)
        job.candidate_version=version; job.status="candidate_ready"; self._write(job); return job
    def load(self,job_id:str)->RetrainingJob:
        return RetrainingJob(**json.loads((self.root/f"{job_id}.json").read_text(encoding="utf-8")))
    def _write(self,job:RetrainingJob)->None:
        (self.root/f"{job.job_id}.json").write_text(json.dumps(job.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
