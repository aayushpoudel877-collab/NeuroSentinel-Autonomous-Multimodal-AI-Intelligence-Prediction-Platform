import pytest
from app.training.config import TrainingConfig

def test_training_config_accepts_defaults():
    TrainingConfig().validate()

def test_training_config_rejects_bad_learning_rate():
    with pytest.raises(ValueError): TrainingConfig(learning_rate=0).validate()
