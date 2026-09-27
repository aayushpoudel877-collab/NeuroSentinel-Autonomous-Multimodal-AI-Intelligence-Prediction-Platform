# Phase 8 — Training & Experimentation Engine

The training layer now provides reproducibility, lazy PyTorch data loading, device resolution, gradient clipping, and a reusable PyTorch trainer.

## Flow

1. Seed the run with `seed_everything`.
2. Build a dataset and `make_loader`.
3. Configure `TrainingConfig`.
4. Construct `TorchTrainer` with a model, optimizer, and criterion.
5. Train and evaluate epochs.
6. Feed validation metrics into `TrainingEngine` for best-model and early-stopping state.
7. Persist checkpoints and experiment metadata through the existing checkpoint/tracker modules.

Heavy dependencies remain optional; the default API does not instantiate PyTorch models.