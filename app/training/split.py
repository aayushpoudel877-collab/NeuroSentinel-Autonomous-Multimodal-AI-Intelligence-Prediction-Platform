import numpy as np

def chronological_split(X,y,train=0.7,validation=0.15):
    n=len(X); a=int(n*train); b=int(n*(train+validation)); return (X[:a],y[:a]),(X[a:b],y[a:b]),(X[b:],y[b:])

def stratified_indices(y,seed=42):
    rng=np.random.default_rng(seed); idx=np.arange(len(y)); rng.shuffle(idx); return idx
