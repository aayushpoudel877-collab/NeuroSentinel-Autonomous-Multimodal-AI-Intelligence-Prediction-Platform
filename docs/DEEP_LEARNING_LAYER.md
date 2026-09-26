# Deep Learning Intelligence Layer

NeuroSentinel now includes trainable modality-specific components while preserving the lightweight runtime path.

- TemporalRegressor: lag-window recursive forecasting.
- ImageFeatureClassifier: trainable image-statistics classification backend.
- LearnedFusion: supervised late-fusion classifier over modality scores and confidences.
- Demo training: `python -m app.training.train_temporal`.

These are research-ready baseline components, not claims of pretrained transformer/CNN/ViT performance. Heavy neural backends remain an explicit next step so the default CI stays small and reproducible.
