from __future__ import annotations
from pathlib import Path
from typing import Any
from app.mlops.gates import PromotionGate
from app.mlops.registry import ModelRegistry
from app.training.retraining import RetrainingManager
from app.training.retraining_pipelines import ForecastRetrainingPipeline
from app.training.scheduler import TrainingScheduler

class TrainingControlPlane:
    """Coordinates executable retraining, candidate evaluation, and scheduling."""

    def __init__(self,registry:ModelRegistry,retraining:RetrainingManager,artifact_root:str|Path="models/artifacts",experiment_root:str|Path="artifacts/experiments",schedule_path:str|Path="artifacts/training/schedules.json")->None:
        self.registry=registry
        self.retraining=retraining
        self.scheduler=TrainingScheduler(schedule_path)
        self.forecast_pipeline=ForecastRetrainingPipeline(artifact_root,experiment_root)

    def run_forecast(self,model:str,values:list[float],window:int=6)->dict[str,Any]:
        record=self.registry.get(model)
        if record.task!="time-series": raise ValueError(f"forecast pipeline is not registered for task: {record.task}")
        job=self.retraining.plan(model,{"retrain":True,"reasons":["manual_or_scheduled_execution"]})
        candidate=self.retraining.execute(job.job_id,lambda current:self.forecast_pipeline.run(current,values,window))
        registered=self.retraining.register_candidate(candidate.job_id)
        return {"job":registered.to_dict(),"candidate":self.registry.get(model,candidate.candidate_version).to_dict()}

    def evaluate_candidate(self,model:str,version:str,metric:str="mae",threshold:float=5.0,greater_is_better:bool=False)->dict[str,Any]:
        record=self.registry.get(model,version)
        if record.status!="candidate":
            raise ValueError(f"only candidate models can be evaluated for staging: {model}:{version}")
        gate=PromotionGate(metric,threshold,greater_is_better)
        passed,reason=gate.evaluate(record.metrics or {})
        result={"model":model,"version":version,"passed":passed,"reason":reason,"metric":metric,"threshold":threshold,"action":"staged" if passed else "rejected"}
        if passed: self.registry.promote(model,version,"staging")
        return result

    def create_schedule(self,model:str,interval_minutes:int,enabled:bool=True)->dict[str,Any]:
        self.registry.get(model)
        return self.scheduler.create(model,interval_minutes,enabled).to_dict()

    def due_schedules(self)->list[dict[str,Any]]:
        return [x for x in self.scheduler.list() if self.scheduler.due(x["schedule_id"])]

    def run_scheduled_forecast(self,schedule_id:str,values:list[float],window:int=6)->dict[str,Any]:
        schedule=self.scheduler.get(schedule_id)
        if not self.scheduler.due(schedule_id): raise ValueError("schedule is not due")
        result=self.run_forecast(schedule.model,values,window)
        self.scheduler.mark_run(schedule_id)
        return {"schedule":self.scheduler.get(schedule_id).to_dict(),"training":result}
