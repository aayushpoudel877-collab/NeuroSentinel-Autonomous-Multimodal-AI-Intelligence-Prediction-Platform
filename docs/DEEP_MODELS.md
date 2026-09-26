# Heavy Neural Model Layer

NeuroSentinel now has an optional deep-learning boundary. PyTorch is intentionally not required by the default dependency set.

## Components

- Optional GELU/LayerNorm MLP backend.
- Optional LSTM temporal backend.
- CPU/CUDA device resolver with safe fallback.
- Local JSON experiment tracker.
- Checkpoint provenance and SHA-256 integrity metadata.

The separation keeps lightweight CI reproducible while allowing research environments to add GPU tooling and heavyweight neural models.