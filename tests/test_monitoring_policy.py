import pytest
from app.monitoring.policy import RetrainingDecision,RetrainingPolicy

def test_retraining_triggers_on_drift():
    result=RetrainingDecision(RetrainingPolicy(drift_threshold=0.1)).evaluate(0.25,100,2)
    assert result["retrain"] is True
    assert "drift_threshold_exceeded" in result["reasons"]

def test_retraining_waits_for_minimum_calls_for_error_signal():
    result=RetrainingDecision(RetrainingPolicy(minimum_calls=100,error_rate_threshold=0.05)).evaluate(0.01,10,10)
    assert result["retrain"] is False

def test_invalid_counts_rejected():
    with pytest.raises(ValueError): RetrainingDecision(RetrainingPolicy()).evaluate(0.1,3,4)
