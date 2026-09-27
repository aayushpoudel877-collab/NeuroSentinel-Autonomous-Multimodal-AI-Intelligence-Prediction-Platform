from app.mlops.registry import ModelRecord,ModelRegistry
from app.serving.router import Route,ServingRouter

def test_router_resolves_registered_handler(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json"); registry.register(ModelRecord("demo","1","classification",status="production")); router=ServingRouter(registry)
    router.register(Route("classification","classification","demo","1"),lambda value:value+1)
    assert router.infer("classification",4)==5
