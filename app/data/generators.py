import numpy as np
def synthetic_series(length=120,seed=42,noise=0.15):
    rng=np.random.default_rng(seed); t=np.arange(length); signal=0.03*t+np.sin(t/5)+0.35*np.sin(t/13)
    return (signal+rng.normal(0,noise,length)).round(5).tolist()
def inject_anomalies(values,indices,magnitude=4.0):
    result=list(values)
    for i in indices:
        if 0<=i<len(result): result[i]+=magnitude
    return result
