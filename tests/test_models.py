from app.models.anomaly import AnomalyDetector
from app.models.forecasting import Forecaster

def test_anomaly_shape():
    result = AnomalyDetector().detect([1,2,3,4,5,6,7,8,9,100])
    assert "anomalies" in result
    assert "scores" in result

def test_forecaster():
    result = Forecaster().predict([1,2,3,4,5,6,7,8,9], 2)
    assert len(result["predictions"]) == 2
