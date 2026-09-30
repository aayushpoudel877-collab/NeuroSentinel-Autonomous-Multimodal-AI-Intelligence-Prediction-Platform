from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from app.mlops.registry import ModelRegistry
@dataclass(frozen=True)
class ChallengerEvaluation:
    model:str; candidate_version:str; champion_version:str|None; metric:str; candidate_value:float|None; champion_value:float|None; passed:bool; reason:str
class ChallengerEvaluator:
    def __init__(self,registry:ModelRegistry)->None: self.registry=registry
    def evaluate(self,model:str,version:str,metric:str,minimum_improvement:float=0.0,greater_is_better:bool=False)->ChallengerEvaluation:
        if minimum_improvement<0: raise ValueError("minimum_improvement must be non-negative")
        candidate=self.registry.get(model,version)
        if candidate.status!="candidate": raise ValueError("challenger must be a candidate model")
        champion=self.registry.production(candidate.task)
        cv=None if not candidate.metrics or metric not in candidate.metrics else float(candidate.metrics[metric])
        hv=None if champion is None or not champion.metrics or metric not in champion.metrics else float(champion.metrics[metric])
        if cv is None: return ChallengerEvaluation(model,version,None if champion is None else champion.version,metric,None,hv,False,f"candidate metric missing: {metric}")
        if champion is None: return ChallengerEvaluation(model,version,None,metric,cv,None,True,"no production champion exists")
        if hv is None: return ChallengerEvaluation(model,version,champion.version,metric,cv,None,False,f"champion metric missing: {metric}")
        improvement=(cv-hv) if greater_is_better else (hv-cv); passed=improvement>=minimum_improvement
        return ChallengerEvaluation(model,version,champion.version,metric,cv,hv,passed,f"candidate {metric}={cv:.6g}, champion={hv:.6g}, improvement={improvement:.6g}; required={minimum_improvement:.6g}")
    def to_dict(self,result:ChallengerEvaluation)->dict[str,Any]: return result.__dict__
