from __future__ import annotations

def resolve_device(preferred: str = 'auto') -> str:
    if preferred not in {'auto','cpu','cuda'}: raise ValueError('preferred must be auto, cpu, or cuda')
    if preferred == 'cpu': return 'cpu'
    try:
        import torch
    except ImportError:
        return 'cpu'
    if preferred == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is unavailable')
    return 'cuda' if torch.cuda.is_available() else 'cpu'
