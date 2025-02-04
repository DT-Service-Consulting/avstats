# core/EDA/validators/validator_FeatureEngineering.py
from pydantic import BaseModel, ConfigDict, Field


class CategorizeFlightInput(BaseModel):
    cargo: bool
    private: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GetTimeWindowInput(BaseModel):
    hour: int = Field(ge=0, le=23, description="Hour must be between 0 and 23")

    model_config = ConfigDict(arbitrary_types_allowed=True)
