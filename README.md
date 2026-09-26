# NeuroSentinel

**Autonomous Multimodal AI Intelligence & Prediction Platform**

NeuroSentinel is a modular ML/DL research platform for combining NLP, computer vision, time-series forecasting, anomaly detection, neural models, confidence calibration, multimodal fusion, evaluation, and monitoring behind one API.

## System layers

Client -> FastAPI API -> Orchestrator -> Model Layer -> Fusion/Monitoring -> Structured response

### Implemented capabilities
- NLP sentiment analysis baseline with reusable text features
- Lagged Ridge time-series forecasting
- Isolation Forest anomaly detection
- Multi-layer MLP neural classifier with training workflow
- Dependency-light image classification baseline
- Confidence-weighted multimodal late fusion
- Distribution-shift/drift monitoring
- Feature validation and rolling statistics
- Experiment registry and evaluation metrics
- Health/model registry endpoints
- CLI demo and automated tests
- Docker and GitHub Actions CI structure

## API

- GET /health
- GET /models
- POST /v1/text/analyze
- POST /v1/forecast
- POST /v1/anomaly
- POST /v1/vision/classify
- POST /v1/fusion
- POST /v1/drift

Interactive API documentation is exposed by FastAPI at /docs when the service is running.

## Local development

Create a Python 3.11 environment, install requirements.txt, then run uvicorn app.main:app --reload. Run pytest -q for the test suite and python -m app.cli demo for the CLI demonstration.

## Research roadmap

The architecture is deliberately model-agnostic. Stronger transformer, CNN/ViT, temporal neural, embedding, retrieval, and domain-specific models can replace the baseline implementations while preserving the orchestration layer.
