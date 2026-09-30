import pytest
from app.training.pipelines import TrainingPipeline,TrainingPipelineRegistry

def test_pipeline_registry_register_and_resolve():
    registry=TrainingPipelineRegistry()
    pipeline=registry.register(TrainingPipeline("demo","task","demo",lambda **_:{"ok":True}))
    assert registry.get("demo") is pipeline
    assert registry.list()[0]["task"]=="task"

def test_pipeline_registry_rejects_duplicate():
    registry=TrainingPipelineRegistry()
    registry.register(TrainingPipeline("demo","task","demo",lambda **_:{}))
    with pytest.raises(ValueError,match="already registered"):
        registry.register(TrainingPipeline("demo","task","demo",lambda **_:{}))
