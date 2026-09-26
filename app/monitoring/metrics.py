from dataclasses import dataclass,field
from time import perf_counter

@dataclass
class InferenceMetrics:
    calls:int=0
    total_seconds:float=0.0
    errors:int=0
    by_model:dict[str,int]=field(default_factory=dict)
    def observe(self,model,elapsed,error=False):
        self.calls+=1; self.total_seconds+=elapsed; self.errors+=int(error); self.by_model[model]=self.by_model.get(model,0)+1
    def snapshot(self):
        return {'calls':self.calls,'errors':self.errors,'average_latency_ms':round((self.total_seconds/max(self.calls,1))*1000,3),'by_model':dict(self.by_model)}
