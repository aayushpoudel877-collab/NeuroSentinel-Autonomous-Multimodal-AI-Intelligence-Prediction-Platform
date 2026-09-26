from __future__ import annotations

"""Optional PyTorch backends with lazy imports."""
from typing import Any

def torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False

def build_mlp(input_dim: int, hidden_dim: int = 128, output_dim: int = 2) -> Any:
    if not torch_available():
        raise RuntimeError("PyTorch is not installed; install the optional deep-learning extra")
    import torch.nn as nn
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.1), nn.Linear(hidden_dim, output_dim))

def build_temporal_lstm(input_dim: int, hidden_dim: int = 64, layers: int = 2) -> Any:
    if not torch_available():
        raise RuntimeError("PyTorch is not installed; install the optional deep-learning extra")
    import torch.nn as nn
    return nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0)
