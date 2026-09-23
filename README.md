# NeuroSentinel

Autonomous Multimodal AI Intelligence & Prediction Platform.

NeuroSentinel is a production-oriented research platform that combines tabular ML, time-series forecasting, NLP, computer vision, anomaly detection, model explainability, experiment tracking, and a unified API.

## Architecture

```
Client -> FastAPI -> Orchestrator
                 -> NLP pipeline
                 -> Vision pipeline
                 -> Forecasting pipeline
                 -> Anomaly detection
                 -> Explainability
                 -> Persistence
```

## Core capabilities

- Multimodal inference orchestration
- Text sentiment/topic analysis
- Image classification with a lightweight vision backbone
- Time-series forecasting with classical ML baselines
- Isolation Forest anomaly detection
- Feature validation and preprocessing
- Confidence-aware prediction responses
- Experiment/run metadata
- Health and model registry endpoints
- Docker and CI-ready project structure

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs.

## Research direction

The system is intentionally modular so individual models can be replaced by stronger pretrained or domain-specific models without changing the API contract.
