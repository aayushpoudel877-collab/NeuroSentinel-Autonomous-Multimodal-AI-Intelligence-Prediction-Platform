import pytest
from app.mlops.gates import PromotionGate,PromotionPolicy
from app.mlops.registry import ModelRecord

def test_promotion_gate_accepts_metric():
    record=ModelRecord("demo","1","classification",metrics={"f1_macro":0.91}); policy=PromotionPolicy({"classification":PromotionGate("f1_macro",0.9)})
    assert policy.check(record)["passed"] is True

def test_promotion_gate_rejects_metric():
    record=ModelRecord("demo","1","classification",metrics={"f1_macro":0.81}); policy=PromotionPolicy({"classification":PromotionGate("f1_macro",0.9)})
    with pytest.raises(ValueError): policy.require(record)
