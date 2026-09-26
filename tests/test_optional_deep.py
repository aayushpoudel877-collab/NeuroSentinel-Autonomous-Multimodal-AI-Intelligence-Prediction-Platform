from app.models.deep_optional import torch_available

def test_optional_backend_does_not_break_core(): assert isinstance(torch_available(), bool)
