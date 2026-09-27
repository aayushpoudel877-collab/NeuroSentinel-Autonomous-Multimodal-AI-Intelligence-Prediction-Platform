import pytest
from app.training.dataloader import make_loader

def test_loader_rejects_invalid_batch_size():
    with pytest.raises(ValueError): make_loader([], batch_size=0)
