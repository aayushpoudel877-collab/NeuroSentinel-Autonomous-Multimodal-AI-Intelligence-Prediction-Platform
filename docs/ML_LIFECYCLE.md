# ML Lifecycle

1. Register dataset metadata.
2. Validate and split data without leakage.
3. Generate reproducible features.
4. Train baseline and neural candidates.
5. Evaluate on untouched validation/test partitions.
6. Record experiment metrics and artifact metadata.
7. Calibrate confidence where appropriate.
8. Register an immutable model name/version with artifact and run lineage.
9. Apply task-specific promotion gates before staging or production.
10. Bind an exact model version to a serving route only when its handler is available.
11. Monitor drift, latency, errors, and per-model traffic.
12. Archive the previous production version when a replacement is promoted.
13. Retrain when monitored evidence justifies a new model version.

The serving router deliberately does not auto-switch to an unbound model after promotion. This avoids a registry state change causing an inference path to point at a handler that has not been loaded and validated.
