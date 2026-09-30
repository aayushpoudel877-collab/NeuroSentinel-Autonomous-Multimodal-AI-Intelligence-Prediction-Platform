from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Callable

@dataclass(frozen=True)
class TrainingPipeline:
    name:str
    task:str
    description:str
    runner:Callable[...,dict[str,Any]]

class TrainingPipelineRegistry:
    """Named in-process registry for safe training-pipeline dispatch."""

    def __init__(self)->None:
        self._pipelines:dict[str,TrainingPipeline]={}

    def register(self,pipeline:TrainingPipeline)->TrainingPipeline:
        if not pipeline.name.strip(): raise ValueError("pipeline name is required")
        if not pipeline.task.strip(): raise ValueError("pipeline task is required")
        if pipeline.name in self._pipelines: raise ValueError(f"pipeline already registered: {pipeline.name}")
        self._pipelines[pipeline.name]=pipeline
        return pipeline

    def get(self,name:str)->TrainingPipeline:
        try: return self._pipelines[name]
        except KeyError as exc: raise KeyError(f"training pipeline not registered: {name}") from exc

    def list(self)->list[dict[str,str]]:
        return [{"name":p.name,"task":p.task,"description":p.description} for p in self._pipelines.values()]
