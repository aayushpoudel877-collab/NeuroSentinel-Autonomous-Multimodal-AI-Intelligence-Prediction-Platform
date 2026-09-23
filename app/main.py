from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import TextRequest, ForecastRequest, AnomalyRequest
from app.orchestrator import NeuroSentinelOrchestrator

app = FastAPI(title="NeuroSentinel", version="0.1.0", description="Multimodal AI intelligence platform")
engine = NeuroSentinelOrchestrator()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "neurosentinel", "version": "0.1.0"}

@app.get("/models")
def models():
    return engine.model_registry()

@app.post("/v1/text/analyze")
def analyze_text(request: TextRequest):
    return engine.text.analyze(request.text)

@app.post("/v1/forecast")
def forecast(request: ForecastRequest):
    if len(request.values) < 8:
        raise HTTPException(422, "At least 8 observations are required")
    return engine.forecast.predict(request.values, request.horizon)

@app.post("/v1/anomaly")
def anomaly(request: AnomalyRequest):
    return engine.anomaly.detect(request.values)

@app.post("/v1/vision/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "An image file is required")
    payload = await file.read()
    return engine.vision.classify(payload)
