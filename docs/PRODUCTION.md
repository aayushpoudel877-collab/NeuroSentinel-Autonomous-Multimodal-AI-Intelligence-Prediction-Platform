# Production Architecture

NeuroSentinel separates API serving, model execution, artifact storage, monitoring, and training concerns. The included production compose file provides a reproducible API container baseline. For larger deployments, place the API behind an ingress/load balancer and move artifacts to object storage.

## Security

Set `NEUROSENTINEL_API_KEY_SHA256` to a SHA-256 digest to enable API-key authentication. Do not commit raw keys. Rotate credentials through the deployment secret manager.

## Observability

`/metrics` exposes structured application metrics. The Prometheus configuration provides a starting point for scraping the service. Add OpenTelemetry and distributed tracing when multiple services are introduced.

## Model lifecycle

Training should produce versioned metrics and metadata, store artifacts outside source control, validate models on held-out data, and promote only after evaluation gates pass.
