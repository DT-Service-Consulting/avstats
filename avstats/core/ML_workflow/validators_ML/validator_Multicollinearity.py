# core/ML_workflow/validators_ML/validator_Multicollinearity.py
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError
import pandas as pd


class MulticollinearityInput(BaseModel):
    scaled_df: pd.DataFrame
    y: pd.Series
    verbose: bool = Field(default=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("scaled_df")
    def validate_scaled_df(cls, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("scaled_df must be a pandas DataFrame.")
        if value.empty:
            raise ValueError("scaled_df cannot be empty.")
        if any(value.isnull().any()):
            raise ValueError("scaled_df contains missing values. Please clean the data before using this class.")
        return value

    @field_validator("y")
    def validate_target(cls, value: pd.Series):
        if not isinstance(value, pd.Series):
            raise ValueError("y must be a pandas Series.")
        if value.empty:
            raise ValueError("y cannot be empty.")
        return value

    @field_validator("verbose")
    def validate_verbose(cls, value: bool):
        if not isinstance(value, bool):
            raise ValueError("verbose must be a boolean value.")
        return value
