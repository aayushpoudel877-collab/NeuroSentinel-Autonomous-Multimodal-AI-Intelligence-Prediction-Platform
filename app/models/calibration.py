import numpy as np
class ConfidenceCalibrator:
    def __init__(self,temperature=1.0): self.temperature=max(temperature,1e-6)
    def calibrate(self,score):
        z=np.clip(float(score)/self.temperature,-30,30); return float(1/(1+np.exp(-z)))
