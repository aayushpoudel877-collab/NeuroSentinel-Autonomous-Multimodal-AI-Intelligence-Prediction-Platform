from app.pipelines.batch import BatchPipeline
from app.data.generators import synthetic_series

def test_batch_pipeline():
    result=BatchPipeline().process_series([synthetic_series(20),[1,2]])
    assert result[0]['valid'] is True
    assert result[1]['valid'] is False
