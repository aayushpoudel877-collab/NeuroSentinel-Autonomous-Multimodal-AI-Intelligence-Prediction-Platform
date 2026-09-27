from fastapi import FastAPI,UploadFile,File,HTTPException,Depends
from app.schemas import TextRequest,ForecastRequest,AnomalyRequest,FusionRequest,DriftRequest,PromotionRequest,RetrainingRequest,RetrainingPlanRequest
from app.orchestrator import NeuroSentinelOrchestrator
from app.security.auth import require_api_key

app=FastAPI(title="NeuroSentinel",version="0.5.0",description="Multimodal AI intelligence and prediction platform")
engine=NeuroSentinelOrchestrator()

@app.get("/health")
def health(): return engine.health()
@app.get("/metrics")
def metrics(): return engine.metrics()
@app.get("/models")
def models(): return engine.model_registry()
@app.get("/v1/registry")
def registry(): return engine.model_registry()
@app.get("/v1/serving")
def serving(): return engine.serving_snapshot()
@app.get("/v1/serving/health")
def serving_health(): return engine.serving_health()
@app.get("/v1/monitoring/events")
def monitoring_events(): return engine.events.recent(100)
@app.post("/v1/models/promote")
def promote(request:PromotionRequest,_=Depends(require_api_key)):
    try: return engine.promote_model(request.model,request.version,request.metric,request.threshold,request.greater_is_better,request.status)
    except KeyError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc
    except ValueError as exc: raise HTTPException(status_code=409,detail=str(exc)) from exc
@app.post("/v1/retraining/evaluate")
def evaluate_retraining(request:RetrainingRequest,_=Depends(require_api_key)):
    try: return engine.evaluate_retraining(request.model,request.reference,request.current,request.recent_retrain_count)
    except KeyError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc
@app.post("/v1/retraining/plan")
def plan_retraining(request:RetrainingPlanRequest,_=Depends(require_api_key)):
    return engine.plan_retraining(request.model,{"retrain":request.retrain,"reasons":request.reasons,"drift_score":request.drift_score,"error_rate":request.error_rate})
@app.get("/v1/retraining/{job_id}")
def get_retraining_job(job_id:str):
    try: return engine.retraining.load(job_id).to_dict()
    except FileNotFoundError as exc: raise HTTPException(status_code=404,detail="retraining job not found") from exc
@app.get("/v1/benchmark")
def benchmark(): return engine.benchmark()
@app.post("/v1/text/analyze")
def analyze_text(request:TextRequest,_=Depends(require_api_key)): return engine.text.analyze(request.text)
@app.post("/v1/forecast")
def forecast(request:ForecastRequest,_=Depends(require_api_key)): return engine.forecast.predict(request.values,request.horizon)
@app.post("/v1/anomaly")
def anomaly(request:AnomalyRequest,_=Depends(require_api_key)): return engine.anomaly.detect(request.values)
@app.post("/v1/fusion")
def fusion(request:FusionRequest,_=Depends(require_api_key)): return engine.fusion.combine(request.signals)
@app.post("/v1/drift")
def drift(request:DriftRequest,_=Depends(require_api_key)): return engine.drift.compare(request.reference,request.current)
@app.post("/v1/vision/classify")
async def classify_image(file:UploadFile=File(...),_=Depends(require_api_key)):
    if not file.content_type or not file.content_type.startswith("image/"): raise HTTPException(415,"An image file is required")
    return engine.vision.classify(await file.read())
