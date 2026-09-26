import numpy as np

from app.models.vision_dl import ImageFeatureClassifier


def test_image_feature_classifier():
    images = [np.zeros((8, 8, 3)), np.ones((8, 8, 3)), np.full((8, 8, 3), 0.2), np.full((8, 8, 3), 0.8)]
    labels = ["dark", "bright", "dark", "bright"]
    model = ImageFeatureClassifier().fit(images, labels)
    assert len(model.predict(images)) == 4
