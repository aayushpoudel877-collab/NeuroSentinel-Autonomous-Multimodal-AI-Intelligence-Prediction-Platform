from __future__ import annotations

import numpy as np

def stratified_indices(labels, test_size: float = 0.2, seed: int = 42):
    if not 0 < test_size < 1: raise ValueError('test_size must be between 0 and 1')
    labels=np.asarray(labels)
    rng=np.random.default_rng(seed)
    train=[]; test=[]
    for label in np.unique(labels):
        indices=np.flatnonzero(labels == label); rng.shuffle(indices)
        cut=max(1, int(round(len(indices)*test_size)))
        if cut >= len(indices) and len(indices) > 1: cut=len(indices)-1
        test.extend(indices[:cut]); train.extend(indices[cut:])
    return np.asarray(sorted(train),dtype=int), np.asarray(sorted(test),dtype=int)
