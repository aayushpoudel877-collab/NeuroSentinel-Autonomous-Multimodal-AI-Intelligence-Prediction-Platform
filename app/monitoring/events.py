from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from pathlib import Path
import json
from typing import Any

@dataclass(frozen=True)
class MonitoringEvent:
    event_id:str
    event_type:str
    model:str
    payload:dict[str,Any]
    created_at:str

class MonitoringEventStore:
    def __init__(self, path:str|Path="artifacts/monitoring/events.jsonl")->None:
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def append(self,event_type:str,model:str,payload:dict[str,Any],event_id:str)->MonitoringEvent:
        event=MonitoringEvent(event_id,event_type,model,payload,datetime.now(timezone.utc).isoformat())
        with self.path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(asdict(event),sort_keys=True)+"\n")
        return event
    def recent(self,limit:int=100)->list[dict[str,Any]]:
        if limit<1: raise ValueError("limit must be positive")
        if not self.path.exists(): return []
        lines=self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
