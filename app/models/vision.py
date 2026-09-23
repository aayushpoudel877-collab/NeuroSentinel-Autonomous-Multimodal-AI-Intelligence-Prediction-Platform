import io
import numpy as np
from PIL import Image

class VisionClassifier:
    """Dependency-light vision baseline; replaceable with a trained CNN/ViT artifact."""
    def classify(self, payload: bytes):
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))
        brightness = float(mean.mean())
        if brightness < 0.2:
            label = "dark_scene"
        elif brightness > 0.8:
            label = "bright_scene"
        else:
            label = "general_scene"
        return {
            "task": "vision_classification",
            "label": label,
            "confidence": round(float(max(brightness, 1 - brightness)), 4),
            "image_size": {"width": image.width, "height": image.height}
        }
