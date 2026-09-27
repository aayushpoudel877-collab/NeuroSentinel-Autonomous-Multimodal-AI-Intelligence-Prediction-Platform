from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Callable
from app.mlops.registry import ModelRegistry

@dataclass
class Route:
    name: str
    task: str
    model: str
    version: str
    enabled: bool = True
    def to_dict(self) -> dict[str,Any]: return asdict(self)

class ServingRouter:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry=registry; self._handlers={}; self._routes={}
    def register(self, route: Route, handler: Callable[...,Any]) -> None:
        self.registry.get(route.model,route.version); self._routes[route.name]=route; self._handlers[(route.model,route.version)]=handler
    def resolve(self, route_name: str) -> tuple[Route,Callable[...,Any]]:
        route=self._routes.get(route_name)
        if route is None or not route.enabled: raise KeyError(f"serving route unavailable: {route_name}")
        handler=self._handlers.get((route.model,route.version))
        if handler is None: raise KeyError(f"handler unavailable for {route.model}:{route.version}")
        return route,handler
    def infer(self, route_name: str,*args:Any,**kwargs:Any)->Any: return self.resolve(route_name)[1](*args,**kwargs)
    def snapshot(self)->dict[str,Any]: return {"routes":[r.to_dict() for r in self._routes.values()]}
