# ML Lifecycle

1. Register dataset metadata.
2. Validate and split data without leakage.
3. Generate reproducible features.
4. Train baseline and neural candidates.
5. Evaluate on untouched validation/test partitions.
6. Record experiment metrics and artifact metadata.
7. Calibrate confidence where appropriate.
8. Monitor drift and inference behavior.
9. Promote models only through explicit evaluation gates.
10. Re-train when monitored evidence justifies a new model version.
