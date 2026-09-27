from __future__ import annotations

from typing import Any


def build_temporal_encoder(input_dim: int, hidden_dim: int = 64, layers: int = 2) -> Any:
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError('PyTorch is required for temporal neural models') from exc
    return nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0)
