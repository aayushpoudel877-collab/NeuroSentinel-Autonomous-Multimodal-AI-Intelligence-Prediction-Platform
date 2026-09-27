import pytest
from app.models.neural_fusion import build_multimodal_fusion

def test_fusion_requires_optional_backend_when_missing():
    try:
        import torch
    except ImportError:
        with pytest.raises(RuntimeError): build_multimodal_fusion(4,4,4)
    else:
        model=build_multimodal_fusion(4,4,4)
        assert model is not None
