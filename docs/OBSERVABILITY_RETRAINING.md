# Phase 11 — Observability, Drift Intelligence & Retraining

NeuroSentinel now has a dependency-light feedback loop from production monitoring to retraining decisions.

## Flow

production inference -> metrics -> drift comparison -> retraining policy -> decision event -> retraining job -> candidate model -> evaluation/promotion

## Monitoring events

MonitoringEventStore appends structured JSONL events containing an event ID, event type, model, payload, and UTC timestamp. Events are intentionally append-only at the application layer so operational decisions remain auditable.

## Retraining policy

A retraining decision can be triggered by:

- drift score reaching the configured threshold;
- error rate reaching the configured threshold after the minimum call volume;
- a cooldown guard preventing repeated retraining decisions from immediately re-queuing work.

The policy does not train a model automatically. It produces an explicit decision and a queued RetrainingJob, keeping model creation and promotion as separate controlled stages.

## API

- GET /v1/monitoring/events — recent monitoring decisions/events.
- POST /v1/retraining/evaluate — compare reference/current data and evaluate the retraining policy.
- POST /v1/retraining/plan — create a persisted retraining job from an approved decision.
- GET /v1/retraining/{job_id} — inspect a retraining job.

## Safety boundary

Automatic retraining is deliberately separated from automatic production promotion. A future scheduler can execute queued jobs, but a candidate must still be evaluated and pass the Phase 10 promotion lifecycle before becoming production.
