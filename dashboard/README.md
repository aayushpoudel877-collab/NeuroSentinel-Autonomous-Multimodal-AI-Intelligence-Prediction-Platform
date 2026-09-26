# NeuroSentinel Dashboard

A dependency-free monitoring dashboard for the FastAPI service.

## Run locally

Start the API with `uvicorn app.main:app --reload`, then serve this directory with any static web server. Set `localStorage.neurosentinel_api` to the API origin when the dashboard is hosted separately.

The dashboard uses the existing health, metrics and model-registry APIs and does not invent telemetry or model results.