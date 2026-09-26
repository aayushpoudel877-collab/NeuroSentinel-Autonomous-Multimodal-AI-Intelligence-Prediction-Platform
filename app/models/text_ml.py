from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TextClassifier:
    """Trainable TF-IDF + logistic-regression text classifier."""

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000)),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ])
        self.fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> "TextClassifier":
        if len(texts) != len(labels) or len(texts) < 2:
            raise ValueError("texts and labels must have equal length and at least two samples")
        self.pipeline.fit(texts, labels)
        self.fitted = True
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if not self.fitted:
            raise RuntimeError("text classifier is not fitted")
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        if not self.fitted:
            raise RuntimeError("text classifier is not fitted")
        return self.pipeline.predict_proba(texts).tolist()
