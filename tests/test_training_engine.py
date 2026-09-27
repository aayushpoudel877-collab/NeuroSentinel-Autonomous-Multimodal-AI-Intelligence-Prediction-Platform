from app.training.config import TrainingConfig
from app.training.engine import TrainingEngine

def test_engine_tracks_best_metric():
    engine=TrainingEngine(TrainingConfig(patience=2))
    assert engine.update(0.7)
    assert not engine.update(0.6)
    assert engine.state.best_metric == 0.7
