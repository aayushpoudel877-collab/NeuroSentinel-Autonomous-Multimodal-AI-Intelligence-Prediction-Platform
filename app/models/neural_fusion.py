from __future__ import annotations

from typing import Any


def build_multimodal_fusion(text_dim: int, vision_dim: int, temporal_dim: int, hidden_dim: int = 128) -> Any:
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError('PyTorch is required for neural multimodal fusion') from exc
    total = text_dim + vision_dim + temporal_dim
    return nn.Sequential(
        nn.Linear(total, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        nn.Linear(hidden_dim // 2, 2)
    )
