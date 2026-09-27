from __future__ import annotations

from typing import Any


def build_image_encoder(num_classes: int = 2) -> Any:
    try:
        import torch.nn as nn
        from torchvision.models import resnet18
    except ImportError as exc:
        raise RuntimeError('PyTorch and torchvision are required for the vision encoder') from exc
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
