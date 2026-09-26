from app.models.text import TextAnalyzer
from app.models.forecasting import Forecaster
from app.models.anomaly import AnomalyDetector
from app.models.vision import VisionClassifier
from app.models.fusion import MultimodalFusion
from app.monitoring.drift import DriftDetector
from app.monitoring.health import HealthMonitor

class NeuroSentinelOrchestrator:
    def __init__(self):
        self.text=TextAnalyzer(); self.forecast=Forecaster(); self.anomaly=AnomalyDetector(); self.vision=VisionClassifier()
        self.fusion=MultimodalFusion(); self.drift=DriftDetector(); self.health_monitor=HealthMonitor()
    def model_registry(self):
        return {"models":[
            {"name":"neurosentinel-text","task":"NLP","status":"ready","version":"0.2.0"},
            {"name":"neurosentinel-forecast","task":"time-series","status":"ready","version":"0.2.0"},
            {"name":"neurosentinel-anomaly","task":"anomaly-detection","status":"ready","version":"0.2.0"},
            {"name":"neurosentinel-neural","task":"multilayer-MLP","status":"trainable","version":"0.2.0"},
            {"name":"neurosentinel-vision","task":"computer-vision","status":"ready","version":"0.2.0"}]}
    def health(self):
        return self.health_monitor.snapshot(len(self.model_registry()["models"]))
