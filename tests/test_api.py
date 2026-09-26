from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_text():
    response = client.post("/v1/text/analyze", json={"text": "great successful improvement"})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"

def test_forecast():
    response = client.post("/v1/forecast", json={"values": [1,2,3,4,5,6,7,8,9,10], "horizon": 3})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


def test_fusion_and_drift():
    fusion=client.post("/v1/fusion",json={"signals":[{"score":0.8,"confidence":0.9},{"score":0.2,"confidence":0.5}]})
    assert fusion.status_code==200
    assert fusion.json()["sources"]==2
    drift=client.post("/v1/drift",json={"reference":[1,2,3,4,5,6,7,8],"current":[2,3,4,5,6,7,8,9]})
    assert drift.status_code==200
    assert "drift_score" in drift.json()
