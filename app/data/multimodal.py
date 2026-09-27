from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np

@dataclass
class MultimodalRecord:
    sample_id: str
    label: int
    text: str | None = None
    vision: np.ndarray | None = None
    temporal: np.ndarray | None = None

    def validate(self) -> None:
        if not self.sample_id: raise ValueError('sample_id is required')
        if self.vision is None and self.temporal is None and self.text is None: raise ValueError('at least one modality is required')
        if self.temporal is not None and not np.isfinite(np.asarray(self.temporal, dtype=float)).all(): raise ValueError('temporal modality contains non-finite values')

class MultimodalDataset:
    def __init__(self, records: list[MultimodalRecord]) -> None:
        if not records: raise ValueError('dataset cannot be empty')
        for record in records: record.validate()
        self.records=records

    def __len__(self) -> int: return len(self.records)
    def labels(self) -> np.ndarray: return np.asarray([r.label for r in self.records])
    def ids(self) -> list[str]: return [r.sample_id for r in self.records]

    def modality_coverage(self) -> dict[str, float]:
        total=len(self.records)
        return {
            'text': sum(r.text is not None for r in self.records)/total,
            'vision': sum(r.vision is not None for r in self.records)/total,
            'temporal': sum(r.temporal is not None for r in self.records)/total,
        }

    def manifest(self) -> dict[str, Any]:
        return {'samples': len(self), 'ids': self.ids(), 'modality_coverage': self.modality_coverage()}
