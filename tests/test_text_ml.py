from app.models.text_ml import TextClassifier


def test_text_classifier_trains_and_predicts():
    model = TextClassifier().fit(["good product", "great service", "bad product", "poor service"], ["pos", "pos", "neg", "neg"])
    assert model.predict(["good service"])[0] in {"pos", "neg"}
    assert len(model.predict_proba(["good service"])[0]) == 2
