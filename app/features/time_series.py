import numpy as np
def lag_matrix(values,lags=7):
    x=np.asarray(values,dtype=float)
    if len(x)<=lags: raise ValueError('not enough observations for requested lags')
    return np.array([x[i-lags:i] for i in range(lags,len(x))]),x[lags:]
def rolling_statistics(values,window=7):
    x=np.asarray(values,dtype=float)
    if len(x)<window: raise ValueError('window exceeds series length')
    tail=x[-window:]
    return {'mean':float(tail.mean()),'std':float(tail.std()),'min':float(tail.min()),'max':float(tail.max()),'trend':float(tail[-1]-tail[0])}
