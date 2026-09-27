from __future__ import annotations

import os
import random
from typing import Any

def seed_everything(seed: int = 42) -> dict[str, Any]:
    if seed < 0: raise ValueError('seed must be non-negative')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return {'seed': seed}
