from app.data.splits import stratified_indices

def test_stratified_split_preserves_classes():
    train,test=stratified_indices([0,0,0,1,1,1],test_size=1/3,seed=1)
    assert set([0,1])==set([ [0,0,0,1,1,1][i] for i in train])
    assert set([0,1])==set([ [0,0,0,1,1,1][i] for i in test])
