from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json
from typing import Any

@dataclass
class CheckpointMetadata:
    model_name: str
    version: str
    epoch: int
    metric_name: str
    metric_value: float
    artifact_sha256: str
    def to_dict(self) -> dict[str, Any]: return asdict(self)

def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()

def write_metadata(path: str | Path, metadata: CheckpointMetadata) -> None:
    target=Path(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(metadata.to_dict(), indent=2)+'\n', encoding='utf-8')
