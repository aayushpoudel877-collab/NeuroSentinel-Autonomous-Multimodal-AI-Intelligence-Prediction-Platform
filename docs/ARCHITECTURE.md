# Architecture

Inputs are validated at the FastAPI boundary and routed by the orchestrator to task-specific model services.

## Model plane
- NLP: transparent lexical baseline, replaceable with a transformer.
- Vision: image-statistics baseline, replaceable with CNN/ViT.
- Forecasting: lagged Ridge regression.
- Anomaly detection: Isolation Forest.

## Production evolution
1. Add object storage and dataset versioning.
2. Add experiment tracking.
3. Replace baselines with trained transformer/CNN/temporal models.
4. Add feature store and model registry.
5. Add drift monitoring and scheduled retraining.
6. Add authentication, rate limiting and observability.
