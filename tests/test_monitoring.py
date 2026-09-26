from app.monitoring.drift import DriftDetector
def test_drift_detector(): assert DriftDetector(0.1).compare([1,2,3,4],[10,11,12,13])['drift_detected'] is True
