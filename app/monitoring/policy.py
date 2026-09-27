from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class RetrainingPolicy:
    drift_threshold:float=0.1
    error_rate_threshold:float=0.05
    minimum_calls:int=100
    cooldown_events:int=10
    def __post_init__(self)->None:
        if self.drift_threshold<0 or self.error_rate_threshold<0: raise ValueError("thresholds must be non-negative")
        if self.minimum_calls<1 or self.cooldown_events<0: raise ValueError("counts must be non-negative and minimum_calls positive")

class RetrainingDecision:
    def __init__(self,policy:RetrainingPolicy): self.policy=policy
    def evaluate(self,drift_score:float,calls:int,errors:int,recent_retrain_count:int=0)->dict[str,Any]:
        if calls<0 or errors<0 or errors>calls: raise ValueError("invalid call/error counts")
        error_rate=errors/max(calls,1)
        reasons=[]
        if drift_score>=self.policy.drift_threshold: reasons.append("drift_threshold_exceeded")
        if calls>=self.policy.minimum_calls and error_rate>=self.policy.error_rate_threshold: reasons.append("error_rate_threshold_exceeded")
        if recent_retrain_count>0 and recent_retrain_count>=self.policy.cooldown_events: reasons.append("cooldown_active")
        eligible=bool(reasons) and "cooldown_active" not in reasons
        return {"retrain":eligible,"reasons":reasons,"drift_score":float(drift_score),"error_rate":float(error_rate),"calls":calls}
