from app.experiments.benchmark import run_tabular_benchmark


def test_tabular_benchmark_has_valid_metrics():
    result = run_tabular_benchmark()
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.latency_ms >= 0.0
