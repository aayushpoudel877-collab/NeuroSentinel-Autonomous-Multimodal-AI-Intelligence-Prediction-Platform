from app.serving.batch import run_batch

def test_batch_serving_accounts_for_item_failures():
    def handler(value):
        if value==2: raise ValueError("bad item")
        return value*2
    result=run_batch("job-1","demo",[1,2,3],handler)
    assert result.processed==2; assert result.failed==1; assert result.outputs==[2,6]
