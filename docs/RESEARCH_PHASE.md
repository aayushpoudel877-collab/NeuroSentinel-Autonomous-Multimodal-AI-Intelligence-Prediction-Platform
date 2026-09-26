# Research Phase

This phase adds three concrete capabilities: a dependency-free operational dashboard, a trainable TF-IDF/logistic-regression NLP baseline, and a reproducible tabular benchmark using scikit-learn's built-in breast-cancer dataset.

## Benchmark contract

`GET /v1/benchmark` executes the benchmark and returns accuracy, macro F1, and per-sample prediction latency. Results are generated at request time rather than stored as fabricated numbers.

## NLP training

`python -m app.training.train_text` trains the small demonstration classifier. Replace its sample corpus with a licensed project dataset for research experiments.

## Next research layer

The next step is to add dataset adapters and optional heavyweight model backends (transformers, CNN/ViT, and temporal neural networks) behind explicit dependencies, while keeping the lightweight CI path reproducible.