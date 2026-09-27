import numpy as np
from app.data.multimodal import MultimodalDataset, MultimodalRecord

def test_multimodal_coverage():
    ds=MultimodalDataset([MultimodalRecord('a',0,text='hello'),MultimodalRecord('b',1,temporal=np.array([1.,2.]))])
    assert len(ds)==2
    assert ds.modality_coverage()['text']==0.5
