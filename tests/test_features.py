from app.data.generators import synthetic_series
from app.features.time_series import rolling_statistics,lag_matrix
def test_synthetic_series(): assert len(synthetic_series(20))==20
def test_feature_shapes():
    X,y=lag_matrix(synthetic_series(20),5); assert X.shape==(15,5) and len(y)==15; assert set(rolling_statistics(synthetic_series(20),5))=={'mean','std','min','max','trend'}
