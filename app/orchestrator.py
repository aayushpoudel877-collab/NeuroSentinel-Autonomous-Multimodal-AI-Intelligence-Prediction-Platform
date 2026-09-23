from app.models.text import TextAnalyzer
from app.models.forecasting import Forecaster
from app.models.anomaly import AnomalyDetector
from app.models.vision import VisionClassifier

class NeuroSentinelOrchestrator:
    def __init__(self):
        self.text = TextAnalyzer()
        self.forecast = Forecaster()
        self.anomaly = AnomalyDetector()
        self.vision = VisionClassifier()

    def model_registry(self):
        return {
            "models": [
                {"name": "neurosentinel-text", "task": "NLP", "status": "ready"},
                {"name": "neurosentinel-forecast", "task": "time-series", "status": "ready"},
                {"name": "neurosentinel-anomaly", "task": "anomaly-detection", "status": "ready"},
                {"name": "neurosentinel-vision", "task": "computer-vision", "status": "ready"},
            ]
        }
