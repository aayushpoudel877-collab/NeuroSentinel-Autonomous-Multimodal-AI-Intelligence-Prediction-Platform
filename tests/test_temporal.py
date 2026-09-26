from app.models.temporal import TemporalRegressor


def test_temporal_regressor_forecasts():
    values = [float(i) for i in range(30)]
    model = TemporalRegressor(lags=5).fit(values)
    predictions = model.forecast(values, horizon=4)
    assert len(predictions) == 4
    assert all(isinstance(x, float) for x in predictions)
