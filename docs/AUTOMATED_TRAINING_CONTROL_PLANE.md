# Phase 14 — Automated Training Control Plane

Phase 14 connects the executable retraining pipeline to a persistent control plane.

## What changed
- Added TrainingScheduler with persistent schedule state and due-time calculation.
- Added TrainingControlPlane for executable forecast retraining.
- Added candidate evaluation using explicit promotion gates.
- A passing candidate gate moves a model to staging; production remains an explicit promotion action.
- Added artifact SHA-256 propagation from the trained file into the retraining job and model registry.
- Added validation for non-finite promotion metrics.
- Added API endpoints for training execution, candidate evaluation, schedules, and due schedules.
- Removed the duplicate retraining-candidate API route from app/main.py.
- Aligned the package version to 0.7.0.

## Lifecycle
monitoring -> retraining decision -> queued job -> running -> train -> validate -> artifact + SHA-256 -> experiment record -> candidate registry -> promotion gate -> staging -> explicit production promotion

## API
- POST /v1/training/forecast/run
- POST /v1/training/candidate/evaluate
- GET /v1/training/schedules
- POST /v1/training/schedules
- GET /v1/training/schedules/due
- POST /v1/training/schedules/{schedule_id}/run

The scheduler is persistent but intentionally does not create an uncontrolled background thread. An external scheduler or worker can call the due endpoint and execute the scheduled job. This keeps deployment behavior deterministic.

## Model promotion
A passing candidate is staged only. Production promotion remains an explicit registry operation, preserving exact-version serving behavior.

## Current executable model
The concrete end-to-end pipeline is the lightweight Ridge time-series forecaster. Transformer, ResNet, GRU, and neural-fusion components remain optional heavy backends and are not represented as executed training jobs without their dependencies and data.

## Verification
The repository was audited through the GitHub source tree. Local execution could not be completed here because outbound GitHub network access is unavailable, so no claim is made that remote CI has passed.