from __future__ import annotations
from typing import Any
from app.mlops.registry import ModelRegistry
from app.serving.router import ServingRouter

def serving_health(registry:ModelRegistry,router:ServingRouter)->dict[str,Any]:
    production=[x for x in registry.list() if x["status"]=="production"]; routes=router.snapshot()["routes"]
    return {"registry_models":len(registry.list()),"production_models":len(production),"routes":len(routes),"ready":bool(routes) and bool(production)}
