from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from typing import Any,Callable,Iterable

@dataclass
class BatchInferenceResult:
    job_id:str; route:str; processed:int; failed:int; outputs:list[Any]; created_at:str
    def to_dict(self)->dict[str,Any]: return asdict(self)

def run_batch(job_id:str,route:str,items:Iterable[Any],handler:Callable[[Any],Any])->BatchInferenceResult:
    outputs=[]; failed=0
    for item in items:
        try: outputs.append(handler(item))
        except Exception: failed+=1
    return BatchInferenceResult(job_id,route,len(outputs),failed,outputs,datetime.now(timezone.utc).isoformat())
