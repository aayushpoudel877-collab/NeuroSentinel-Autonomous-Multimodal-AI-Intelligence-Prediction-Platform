from __future__ import annotations

from sklearn.metrics import classification_report

from app.models.text_ml import TextClassifier


def train_demo_classifier() -> tuple[TextClassifier, dict]:
    texts = [
        "excellent service and helpful support", "fast delivery and great quality",
        "I love this product", "very happy with the experience",
        "terrible service and poor support", "delivery was late and disappointing",
        "I dislike this product", "very unhappy with the experience",
    ]
    labels = ["positive", "positive", "positive", "positive", "negative", "negative", "negative", "negative"]
    model = TextClassifier().fit(texts, labels)
    predictions = model.predict(texts)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return model, report


if __name__ == "__main__":
    _, report = train_demo_classifier()
    print(report)
