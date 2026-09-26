from dataclasses import dataclass,asdict
from datetime import datetime,timezone
@dataclass
class Experiment:
    name:str; model:str; metric:float; metric_name:str; created_at:str
class ExperimentRegistry:
    def __init__(self): self._items=[]
    def log(self,name,model,metric,metric_name='score'):
        item=Experiment(name,model,float(metric),metric_name,datetime.now(timezone.utc).isoformat()); self._items.append(item); return asdict(item)
    def list(self): return [asdict(x) for x in self._items]
