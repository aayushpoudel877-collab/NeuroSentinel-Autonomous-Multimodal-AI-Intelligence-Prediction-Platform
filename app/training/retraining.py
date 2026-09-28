from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
import uuid
from pathlib import Path
import json
from typing import Any,Callable
from app.mlops.registry import ModelRecord,ModelRegistry

VALID_JOB_STATES={"queued","running","candidate_ready","failed","not_required"}

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
    run_id:str|None=None
    artifact_uri:str|None=None
    metrics:dict[str,float]|None=None
    error:str|None=None
    finished_at:str|None=None
    def to_dict(self)->dict[str,Any]: return asdict(self)

class RetrainingManager:
    def __init__(self,registry:ModelRegistry,root:str|Path="artifacts/retraining")->None:
        self.registry=registry; self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)

    def plan(self,model:str,decision:dict[str,Any])->RetrainingJob:
        record=self.registry.get(model); now=datetime.now(timezone.utc).isoformat()
        job_id=f"retrain-{uuid.uuid4().hex[:12]}"
        status="queued" if decision.get("retrain") else "not_required"
        job=RetrainingJob(job_id,model,record.task,decision.get("reasons",[]),status,now,record.version)
        self._write(job); return job

    def start(self,job_id:str)->RetrainingJob:
        job=self.load(job_id)
        if job.status!="queued": raise ValueError(f"job cannot start from state: {job.status}")
        job.status="running"; self._write(job); return job

    def fail(self,job_id:str,error:str)->RetrainingJob:
        job=self.load(job_id)
        job.status="failed"; job.error=error; job.finished_at=datetime.now(timezone.utc).isoformat()
        self._write(job); return job

    def mark_candidate(self,job_id:str,version:str,artifact_uri:str|None=None,metrics:dict[str,float]|None=None,run_id:str|None=None)->RetrainingJob:
        job=self.load(job_id)
        if job.status!="running": raise ValueError(f"candidate cannot be recorded from state: {job.status}")
        job.candidate_version=version; job.artifact_uri=artifact_uri; job.metrics=metrics or {}; job.run_id=run_id
        job.status="candidate_ready"; job.finished_at=datetime.now(timezone.utc).isoformat()
        self._write(job); return job

    def register_candidate(self,job_id:str,artifact_uri:str|None=None,metrics:dict[str,float]|None=None,run_id:str|None=None)->RetrainingJob:
        job=self.load(job_id)
        if job.status!="candidate_ready" or not job.candidate_version:
            raise ValueError("job must have a candidate version before registry registration")
        source=self.registry.get(job.model,job.source_version) if job.source_version else self.registry.get(job.model)
        self.registry.register(ModelRecord(
            name=job.model, version=job.candidate_version, task=job.task, status="candidate",
            artifact_uri=artifact_uri or job.artifact_uri, metrics=metrics or job.metrics or {},
            run_id=run_id or job.run_id, parent_version=source.version
        ))
        return job

    def execute(self,job_id:str,trainer:Callable[[RetrainingJob],dict[str,Any]])->RetrainingJob:
        job=self.start(job_id)
        try:
            result=trainer(job)
            version=str(result["version"])
            return self.mark_candidate(job.job_id,version,result.get("artifact_uri"),result.get("metrics"),result.get("run_id"))
        except Exception as exc:
            self.fail(job.job_id,str(exc))
            raise

    def load(self,job_id:str)->RetrainingJob:
        return RetrainingJob(**json.loads((self.root/f"{job_id}.json").read_text(encoding="utf-8")))

    def _write(self,job:RetrainingJob)->None:
        if job.status not in VALID_JOB_STATES: raise ValueError(f"invalid retraining state: {job.status}")
        tmp=self.root/f"{job.job_id}.json.tmp"
        tmp.write_text(json.dumps(job.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
        tmp.replace(self.root/f"{job.job_id}.json")
