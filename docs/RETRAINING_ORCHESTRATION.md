# Retraining Orchestration

NeuroSentinel now exposes a durable retraining control plane.

## Lifecycle

`monitoring -> decision -> queued -> running -> candidate_ready -> candidate registry -> promotion gate -> production`

A failed execution moves the job to `failed`. A decision that does not require retraining is recorded as `not_required`.

## Job state

Each job is persisted as JSON under `artifacts/retraining/` and records the source version, candidate version, run identifier, artifact URI, metrics, reasons, and failure/timing metadata.

## Trainer integration

`RetrainingManager.execute()` accepts a model-specific trainer callable. The trainer returns a candidate version and may return artifact URI, metrics, and run ID. This keeps orchestration separate from optional heavyweight training.

## Candidate registration

Candidates enter the model registry as `candidate` with `parent_version` lineage. Registration never automatically promotes a model to production.

## API

- `POST /v1/retraining/evaluate`
- `POST /v1/retraining/plan`
- `POST /v1/retraining/{job_id}/start`
- `POST /v1/retraining/{job_id}/candidate`
- `POST /v1/retraining/{job_id}/fail`
- `GET /v1/retraining/{job_id}`

All operational retraining endpoints require the configured API key.

## Boundary

This phase implements orchestration and lifecycle management. It does not fabricate training results. Actual training still requires datasets, feature pipelines, model implementations, compute, evaluation, and artifact storage.
