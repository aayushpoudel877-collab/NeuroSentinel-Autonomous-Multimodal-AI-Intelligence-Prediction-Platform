from __future__ import annotations

from typing import Any

def make_loader(dataset: Any, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> Any:
    if batch_size < 1: raise ValueError('batch_size must be positive')
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError('PyTorch is required for neural data loading') from exc
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
