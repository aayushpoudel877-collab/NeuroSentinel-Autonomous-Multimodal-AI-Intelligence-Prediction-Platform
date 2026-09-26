from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

@dataclass
class ExperimentRun:
    run_id: str
    model: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    created_at: str
    def to_dict(self) -> dict[str, Any]: return asdict(self)

class ExperimentTracker:
    def __init__(self, root: str | Path='artifacts/experiments') -> None:
        self.root=Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def log(self, run_id: str, model: str, parameters: dict[str, Any], metrics: dict[str, float]) -> Path:
        run=ExperimentRun(run_id,model,parameters,metrics,datetime.now(timezone.utc).isoformat())
        path=self.root/f'{run_id}.json'; path.write_text(json.dumps(run.to_dict(),indent=2)+'\n',encoding='utf-8'); return path
    def load(self, run_id: str) -> dict[str, Any]: return json.loads((self.root/f'{run_id}.json').read_text(encoding='utf-8'))
