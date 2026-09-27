# Multimodal Deep Learning

Phase 7 introduces the actual neural model boundary for NeuroSentinel.

## Encoders

- Transformer text encoder with lazy Hugging Face loading.
- ResNet vision encoder with a replaceable classification head.
- GRU temporal encoder for sequential signals.
- Neural late-fusion head combining learned modality representations.

## Training

`TrainingConfig` centralizes optimizer and training-loop controls, while `TrainingEngine` tracks best validation metrics and early-stopping state.

Heavy model weights are never downloaded during module import. This keeps the default application and CI path lightweight and deterministic.
