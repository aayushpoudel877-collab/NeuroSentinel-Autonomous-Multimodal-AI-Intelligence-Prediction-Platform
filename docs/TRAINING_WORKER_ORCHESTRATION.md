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
