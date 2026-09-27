from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
import json
from pathlib import Path
from typing import Any

@dataclass
class ModelRecord:
    name:str
    version:str
    task:str
    status:str="candidate"
    artifact_uri:str|None=None
    artifact_sha256:str|None=None
    metrics:dict[str,float]|None=None
    run_id:str|None=None
    created_at:str=""
    promoted_at:str|None=None
    parent_version:str|None=None
    def to_dict(self)->dict[str,Any]: return asdict(self)

class ModelRegistry:
    VALID_STATES={"candidate","staging","production","archived"}
    def __init__(self,path:str|Path="models/registry.json")->None:
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._records=[]; self._load()
    def _load(self)->None:
        if self.path.exists(): self._records=[ModelRecord(**item) for item in json.loads(self.path.read_text(encoding="utf-8")).get("models",[])]
    def _save(self)->None:
        payload={"models":[r.to_dict() for r in self._records]}; tmp=self.path.with_suffix(self.path.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); tmp.replace(self.path)
    def register(self,record:ModelRecord)->ModelRecord:
        if record.status not in self.VALID_STATES: raise ValueError(f"invalid model status: {record.status}")
        if not record.name or not record.version or not record.task: raise ValueError("model name, version, and task are required")
        if not record.created_at: record.created_at=datetime.now(timezone.utc).isoformat()
        self._records=[r for r in self._records if not(r.name==record.name and r.version==record.version)]; self._records.append(record); self._save(); return record
    def list(self,task:str|None=None)->list[dict[str,Any]]:
        records=self._records if task is None else [r for r in self._records if r.task==task]; return [r.to_dict() for r in records]
    def get(self,name:str,version:str|None=None)->ModelRecord:
        matches=[r for r in self._records if r.name==name and(version is None or r.version==version)]
        if not matches: raise KeyError(f"model not registered: {name}:{version or '*'}")
        return matches[-1] if version is None else matches[0]
    def production(self,task:str)->ModelRecord|None:
        for record in reversed(self._records):
            if record.task==task and record.status=="production": return record
        return None
    def promote(self,name:str,version:str,status:str)->ModelRecord:
        if status not in {"staging","production","archived"}: raise ValueError("promotion status must be staging, production, or archived")
        target=self.get(name,version)
        if status=="production":
            for record in self._records:
                if record.task==target.task and record.status=="production": record.status="archived"
            target.promoted_at=datetime.now(timezone.utc).isoformat()
        target.status=status; self._save(); return target
    def snapshot(self)->dict[str,Any]: return {"count":len(self._records),"models":self.list()}
