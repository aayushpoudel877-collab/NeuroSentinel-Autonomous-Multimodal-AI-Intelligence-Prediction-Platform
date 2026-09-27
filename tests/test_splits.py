import pytest
from app.data.splits import stratified_indices

def test_stratified_split_preserves_classes():
    labels=[0,0,0,1,1,1]; train,test=stratified_indices(labels,test_size=1/3,seed=1)
    assert set(labels[i] for i in train)=={0,1}; assert set(labels[i] for i in test)=={0,1}

def test_stratified_split_rejects_singleton_class():
    with pytest.raises(ValueError,match="at least 2"):
        stratified_indices([0,0,1],test_size=0.2)
