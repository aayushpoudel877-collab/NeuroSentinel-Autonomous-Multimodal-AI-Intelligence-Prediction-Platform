import hashlib
import pytest
from app.mlops.registry import ModelRecord,ModelRegistry

def test_registry_rejects_missing_artifact(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json")
    with pytest.raises(ValueError,match="artifact does not exist"):
        registry.register(ModelRecord("m","1","task",artifact_uri=str(tmp_path/"missing.bin")))

def test_registry_rejects_bad_artifact_hash(tmp_path):
    artifact=tmp_path/"model.bin"; artifact.write_bytes(b"model")
    registry=ModelRegistry(tmp_path/"registry.json")
    with pytest.raises(ValueError,match="sha256"):
        registry.register(ModelRecord("m","1","task",artifact_uri=str(artifact),artifact_sha256="bad"))

def test_registry_accepts_valid_artifact(tmp_path):
    artifact=tmp_path/"model.bin"; artifact.write_bytes(b"model")
    digest=hashlib.sha256(b"model").hexdigest()
    registry=ModelRegistry(tmp_path/"registry.json")
    record=registry.register(ModelRecord("m","1","task",artifact_uri=str(artifact),artifact_sha256=digest,metrics={"mae":1.0}))
    assert record.artifact_sha256==digest
