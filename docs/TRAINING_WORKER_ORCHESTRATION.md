# Phase 15 — Worker-Safe Automated Training Orchestration

Phase 15 upgrades the Phase 14 scheduler into a worker-oriented execution layer.

## Fixes from Phase 14

- Fixed FastAPI route ordering so `GET /v1/training/schedules/due` is matched before `/v1/training/schedules/{schedule_id}/run`.
- Added persistent schedule execution metadata.
- Added failure counters and last-job lineage.
- Added lease expiry so abandoned work can become eligible again.
- Added atomic filesystem locking and state reload during lease acquisition to prevent duplicate claims across worker processes on the same host.
- Preserved candidate-only staging so production models are never silently replaced.

## New worker layer

`TrainingWorker`:

1. discovers due schedules;
2. atomically claims one with a lease;
3. invokes the registered training callback;
4. requires a resulting training job ID;
5. records successful completion;
6. releases the lease and increments failure count when execution fails.

## Execution flow

```
persistent schedule
      ↓
due schedule
      ↓
atomic lease claim
      ↓
training worker
      ↓
retraining job
      ↓
candidate artifact
      ↓
registry candidate
      ↓
promotion gate
      ↓
staging
      ↓
explicit production promotion
```

The scheduler remains deliberately pull-based rather than starting an uncontrolled background thread. Kubernetes CronJobs, systemd timers, Celery/RQ workers, or another deployment scheduler can call the API/worker layer later.

## Important limitation

The current concrete training implementation remains the lightweight Ridge forecasting pipeline. Heavy Transformer/ResNet/GRU/neural-fusion training remains optional and is not claimed as executed.

## Verification

GitHub source inspection was completed after the changes. Remote workflow results remain unavailable for the repository's latest commits, so CI success is not claimed without an actual workflow run.


## Phase 16 — Pipeline Registry & Artifact Integrity

Phase 16 adds a named training-pipeline registry and hardens model registration.

- `TrainingPipelineRegistry` provides explicit named dispatch instead of coupling workers to one concrete trainer.
- The forecast Ridge pipeline is registered as `forecast-ridge`.
- `GET /v1/training/pipelines` exposes the registered pipeline catalog.
- Registry registration validates metric finiteness.
- When an artifact URI is supplied, the registry requires the file to exist.
- When an artifact SHA-256 is supplied, the registry recomputes the artifact digest and rejects mismatches.
- Schedule mutations now use the same filesystem lock used by lease claims and reload persistent state before mutation.
- Worker failure cleanup no longer masks the original training exception if lease cleanup itself fails.

The platform version is now `0.9.0`.

Production promotion is still explicit and gate-controlled; successful training only creates a candidate and candidate evaluation can move it to staging.


## Phase 17 — Champion/Challenger Evaluation

Phase 17 adds a read-only challenger evaluator that compares a candidate model with the current production champion for the same task.

- Missing candidate or champion metrics fail safely.
- Lower-is-better and higher-is-better metrics are supported.
- A minimum improvement margin can be required.
- Challenger evaluation does not mutate registry state.
- Explicit promotion remains a separate operation.

Platform version: `1.0.0`.
