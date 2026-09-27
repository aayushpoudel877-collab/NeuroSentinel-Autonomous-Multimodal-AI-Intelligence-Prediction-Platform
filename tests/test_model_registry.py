from app.mlops.registry import ModelRecord,ModelRegistry

def test_registry_promotes_one_production_model(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json"); registry.register(ModelRecord("demo","1","classification",metrics={"f1_macro":0.9})); registry.promote("demo","1","production")
    assert registry.production("classification").version=="1"

def test_registry_replaces_previous_production(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json"); registry.register(ModelRecord("demo","1","classification")); registry.register(ModelRecord("demo","2","classification")); registry.promote("demo","1","production"); registry.promote("demo","2","production")
    assert [r["status"] for r in registry.list("classification")]==["archived","production"]
