from __future__ import annotations

from pathlib import Path
import json
from typing import Any

def write_evaluation_report(report: dict[str, Any], path: str | Path) -> str:
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return str(target)
