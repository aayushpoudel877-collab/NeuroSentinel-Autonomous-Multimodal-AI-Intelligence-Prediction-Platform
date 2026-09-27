from app.training.config import TrainingConfig
from app.training.engine import TrainingEngine

def test_min_mode_improves_when_loss_decreases():
    engine=TrainingEngine(TrainingConfig(patience=2),mode='min')
    assert engine.update(1.0)
    assert engine.update(0.5)
