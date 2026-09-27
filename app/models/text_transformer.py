from __future__ import annotations

from typing import Any

class TransformerTextEncoder:
    def __init__(self, model_name: str = 'distilbert-base-uncased') -> None:
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('Transformers is required for the transformer text encoder') from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def encode(self, texts: list[str]) -> Any:
        if self.model is None or self.tokenizer is None:
            self.load()
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]
