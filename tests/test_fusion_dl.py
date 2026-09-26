from app.models.fusion_dl import LearnedFusion


def test_learned_fusion_predicts():
    features = [[1, 0.9, 0.2, 0.8], [0.1, 0.8, 1, 0.7], [0.9, 0.8, 0.1, 0.9], [0.2, 0.7, 0.9, 0.8]]
    labels = ["a", "b", "a", "b"]
    model = LearnedFusion().fit(features, labels)
    assert model.predict([[0.8, 0.9, 0.2, 0.8]])[0] in {"a", "b"}
