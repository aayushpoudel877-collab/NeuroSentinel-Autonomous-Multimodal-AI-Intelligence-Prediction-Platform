# Production MLOps and Serving Layer

Phase 10 introduces an explicit lifecycle between trained artifacts and online inference.

## Components

- Model registry: file-backed model/version records with candidate, staging, production, and archived states.
- Promotion policy: task-specific evaluation gates prevent unqualified candidates from being promoted.
- Serving router: maps stable route names to exact model versions and handlers.
- Batch inference: reusable job wrapper for batch scoring with per-item failure accounting.
- Serving health: summarizes registry and route readiness.

## Lifecycle

dataset -> train -> evaluate -> artifact -> register -> gate -> staging -> production -> route -> monitor

Promotion is explicit. A new production model archives the previous production version for the same task.

## Operational boundary

The registry and serving layer are dependency-light. Heavy PyTorch/Transformers models remain optional and can be attached through the same exact-version route interface without forcing heavyweight dependencies into the base API image.
