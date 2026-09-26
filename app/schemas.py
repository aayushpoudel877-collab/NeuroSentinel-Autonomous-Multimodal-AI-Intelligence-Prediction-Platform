from pydantic import BaseModel, Field
class TextRequest(BaseModel): text: str=Field(min_length=1,max_length=10000)
class ForecastRequest(BaseModel): values:list[float]=Field(min_length=8); horizon:int=Field(default=5,ge=1,le=100)
class AnomalyRequest(BaseModel): values:list[float]=Field(min_length=8)
class FusionSignal(BaseModel): score:float; confidence:float=Field(default=1.0,ge=0,le=1)
class FusionRequest(BaseModel): signals:list[FusionSignal]=Field(min_length=1,max_length=32)
class DriftRequest(BaseModel): reference:list[float]=Field(min_length=8); current:list[float]=Field(min_length=8)
