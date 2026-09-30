import pytest
from app.mlops.challenger import ChallengerEvaluator
from app.mlops.registry import ModelRecord,ModelRegistry
def test_candidate_beats_champion(tmp_path):
    r=ModelRegistry(tmp_path/"registry.json"); r.register(ModelRecord("m","1","time-series","production",metrics={"mae":5.0})); r.register(ModelRecord("m","2","time-series","candidate",metrics={"mae":4.0}))
    x=ChallengerEvaluator(r).evaluate("m","2","mae",0.5); assert x.passed and x.champion_version=="1"
def test_missing_metric_fails(tmp_path):
    r=ModelRegistry(tmp_path/"registry.json"); r.register(ModelRecord("m","1","task","production",metrics={"mae":5.0})); r.register(ModelRecord("m","2","task","candidate",metrics={}))
    assert not ChallengerEvaluator(r).evaluate("m","2","mae").passed
def test_non_candidate_rejected(tmp_path):
    r=ModelRegistry(tmp_path/"registry.json"); r.register(ModelRecord("m","1","task","production",metrics={"mae":1.0}))
    with pytest.raises(ValueError,match="candidate"): ChallengerEvaluator(r).evaluate("m","1","mae")
