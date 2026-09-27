from app.models.text import TextAnalyzer
from app.models.forecasting import Forecaster
from app.models.anomaly import AnomalyDetector
from app.models.vision import VisionClassifier
from app.models.fusion import MultimodalFusion
from app.monitoring.drift import DriftDetector
from app.monitoring.health import HealthMonitor
from app.monitoring.metrics import InferenceMetrics
from app.experiments.benchmark import run_tabular_benchmark
from app.models.temporal import TemporalRegressor
from app.models.fusion_dl import LearnedFusion
from app.models.vision_dl import ImageFeatureClassifier
from app.mlops.registry import ModelRecord,ModelRegistry
from app.mlops.gates import PromotionGate,PromotionPolicy
from app.serving.router import Route,ServingRouter
from app.serving.health import serving_health

class NeuroSentinelOrchestrator:
    def __init__(self):
        self.text=TextAnalyzer(); self.forecast=Forecaster(); self.anomaly=AnomalyDetector(); self.vision=VisionClassifier(); self.fusion=MultimodalFusion(); self.temporal=TemporalRegressor(); self.learned_fusion=LearnedFusion(); self.vision_dl=ImageFeatureClassifier(); self.drift=DriftDetector(); self.health_monitor=HealthMonitor(); self.inference_metrics=InferenceMetrics()
        self.registry=ModelRegistry(); self.serving=ServingRouter(self.registry); self._seed_registry(); self._register_routes()
    def _seed_registry(self):
        models=[
            ("neurosentinel-text","NLP","production"),("neurosentinel-forecast","time-series","production"),("neurosentinel-anomaly","anomaly-detection","production"),
            ("neurosentinel-neural","multilayer-MLP","candidate"),("neurosentinel-vision","computer-vision","production"),("neurosentinel-embedding","text-embedding","production"),
            ("neurosentinel-temporal","temporal-regression","candidate"),("neurosentinel-learned-fusion","multimodal-fusion","candidate"),("neurosentinel-vision-dl","vision-feature-classification","candidate"),
            ("neurosentinel-transformer-text","transformer-NLP","candidate"),("neurosentinel-resnet-vision","neural-vision","candidate"),("neurosentinel-gru-temporal","neural-temporal","candidate"),("neurosentinel-neural-fusion","neural-multimodal-fusion","candidate")]
        existing={(x["name"],x["version"]) for x in self.registry.list()}
        for name,task,status in models:
            if (name,"0.5.0") not in existing: self.registry.register(ModelRecord(name,"0.5.0",task,status=status))
    def _register_routes(self):
        routes=[
            Route("text","NLP","neurosentinel-text","0.5.0"),Route("forecast","time-series","neurosentinel-forecast","0.5.0"),Route("anomaly","anomaly-detection","neurosentinel-anomaly","0.5.0"),Route("fusion","multimodal-fusion","neurosentinel-learned-fusion","0.5.0",enabled=False),Route("vision","computer-vision","neurosentinel-vision","0.5.0")]
        handlers={"text":self.text.analyze,"forecast":lambda values,horizon:self.forecast.predict(values,horizon),"anomaly":self.anomaly.detect,"fusion":self.fusion.combine,"vision":self.vision.classify}
        for route in routes:
            if route.name=="fusion": continue
            self.serving.register(route,handlers[route.name])
    def model_registry(self): return self.registry.snapshot()
    def serving_snapshot(self): return self.serving.snapshot()
    def serving_health(self): return serving_health(self.registry,self.serving)
    def promote_model(self,name,version,metric,threshold,greater_is_better=True,status="production"):
        record=self.registry.get(name,version); policy=PromotionPolicy({record.task:PromotionGate(metric,threshold,greater_is_better)}); policy.require(record)
        return self.registry.promote(name,version,status).to_dict()
    def health(self):
        base=self.health_monitor.snapshot(len(self.registry.list())); base["serving"]=self.serving_health(); return base
    def metrics(self): return self.inference_metrics.snapshot()
    def benchmark(self): return run_tabular_benchmark().to_dict()
