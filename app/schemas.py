from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

class ForecastRequest(BaseModel):
    values: list[float] = Field(min_length=8)
    horizon: int = Field(default=5, ge=1, le=100)

class AnomalyRequest(BaseModel):
    values: list[float] = Field(min_length=8)
