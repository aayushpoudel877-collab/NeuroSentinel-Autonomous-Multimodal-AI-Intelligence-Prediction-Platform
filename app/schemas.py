from pydantic import BaseModel,Field
class TextRequest(BaseModel): text:str=Field(min_length=1,max_length=10000)
class ForecastRequest(BaseModel): values:list[float]=Field(min_length=8); horizon:int=Field(default=5,ge=1,le=100)
class AnomalyRequest(BaseModel): values:list[float]=Field(min_length=8)
class FusionSignal(BaseModel): score:float; confidence:float=Field(default=1.0,ge=0,le=1)
class FusionRequest(BaseModel): signals:list[FusionSignal]=Field(min_length=1,max_length=32)
class DriftRequest(BaseModel): reference:list[float]=Field(min_length=8); current:list[float]=Field(min_length=8)
class PromotionRequest(BaseModel):
    model:str=Field(min_length=1,max_length=200); version:str=Field(min_length=1,max_length=50); metric:str=Field(min_length=1,max_length=100); threshold:float; greater_is_better:bool=True; status:str=Field(default="production",pattern="^(staging|production)$")
class RetrainingRequest(BaseModel):
    model:str=Field(min_length=1,max_length=200); reference:list[float]=Field(min_length=8); current:list[float]=Field(min_length=8); recent_retrain_count:int=Field(default=0,ge=0)
class RetrainingPlanRequest(BaseModel):
    model:str=Field(min_length=1,max_length=200); retrain:bool; reasons:list[str]=Field(default_factory=list); drift_score:float=Field(default=0.0,ge=0); error_rate:float=Field(default=0.0,ge=0)
