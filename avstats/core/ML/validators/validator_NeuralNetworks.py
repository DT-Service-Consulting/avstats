# core/ML/validators/validator_NeuralNetworks.py
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd


class NeuralNetworksInput(BaseModel):
    df: pd.DataFrame = Field(..., description="DataFrame containing the time series data.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
