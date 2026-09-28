# Automated Training Execution

Phase 13 connects the retraining control plane to a concrete train/validate/artifact workflow.

## Current executable pipeline

`ForecastRetrainingPipeline` implements a lightweight Ridge time-series retraining path:

1. validates the incoming series
2. constructs lag features
3. performs a chronological train/validation split
4. trains Ridge
5. computes MAE and RMSE on the validation segment
6. writes a serialized model artifact
7. records an experiment run
8. returns a unique candidate version

The pipeline can be invoked through `RetrainingManager.execute()` or `execute_and_register()`.

## Lifecycle

`decision -> queued -> running -> train -> validate -> artifact -> experiment -> candidate -> registry`

Production promotion remains a separate gated action.

## Heavy models

Transformer, ResNet, GRU, and neural-fusion models remain optional. Their execution should use the same trainer contract once datasets, compute, and optional dependencies are available. No heavyweight training result is claimed by this phase.

## Reproducibility

The existing training infrastructure remains responsible for device selection, configuration, seeding, checkpoints, and experiment tracking. The new pipeline deliberately keeps its data interface explicit so a scheduler or dataset service can invoke it later.
