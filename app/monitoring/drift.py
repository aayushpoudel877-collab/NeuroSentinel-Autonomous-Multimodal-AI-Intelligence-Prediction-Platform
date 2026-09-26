import numpy as np
class DriftDetector:
    def __init__(self,threshold=0.1): self.threshold=threshold
    def compare(self,reference,current):
        a=np.asarray(reference,dtype=float); b=np.asarray(current,dtype=float)
        if len(a)==0 or len(b)==0: raise ValueError('both distributions require data')
        mean_shift=abs(float(a.mean()-b.mean()))/(float(a.std())+1e-8); variance_shift=abs(float(a.var()-b.var()))/(float(a.var())+1e-8); drift=max(mean_shift,variance_shift)
        return {'drift_score':round(drift,6),'drift_detected':bool(drift>self.threshold),'threshold':self.threshold}
