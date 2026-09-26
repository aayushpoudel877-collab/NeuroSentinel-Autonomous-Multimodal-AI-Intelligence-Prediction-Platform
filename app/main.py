from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from app.schemas import TextRequest, ForecastRequest, AnomalyRequest, FusionRequest, DriftRequest
from app.orchestrator import NeuroSentinelOrchestrator
from app.security.auth import require_api_key

app=FastAPI(title='NeuroSentinel',version='0.3.0',description='Multimodal AI intelligence and prediction platform')
engine=NeuroSentinelOrchestrator()

@app.get('/health')
def health(): return engine.health()
@app.get('/metrics')
def metrics(): return engine.metrics()
@app.get('/models')
def models(): return engine.model_registry()
@app.post('/v1/text/analyze')
def analyze_text(request:TextRequest,_=Depends(require_api_key)): return engine.text.analyze(request.text)
@app.post('/v1/forecast')
def forecast(request:ForecastRequest,_=Depends(require_api_key)): return engine.forecast.predict(request.values,request.horizon)
@app.post('/v1/anomaly')
def anomaly(request:AnomalyRequest,_=Depends(require_api_key)): return engine.anomaly.detect(request.values)
@app.post('/v1/fusion')
def fusion(request:FusionRequest,_=Depends(require_api_key)): return engine.fusion.combine(request.signals)
@app.post('/v1/drift')
def drift(request:DriftRequest,_=Depends(require_api_key)): return engine.drift.compare(request.reference,request.current)
@app.post('/v1/vision/classify')
async def classify_image(file:UploadFile=File(...),_=Depends(require_api_key)):
    if not file.content_type or not file.content_type.startswith('image/'): raise HTTPException(415,'An image file is required')
    return engine.vision.classify(await file.read())
