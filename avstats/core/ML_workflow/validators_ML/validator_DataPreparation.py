# core/ML_workflow/validators_ML/validator_DataPreparation.py
from pydantic import BaseModel, Field, field_validator, ConfigDict
import pandas as pd


class DataPreparationInput(BaseModel):
    df: pd.DataFrame
    target_variable: str = Field(..., description="The name of the target variable column")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df", mode="before")
    def validate_dataframe(cls, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        return df

    @field_validator("target_variable")
    def validate_target_variable(cls, target_variable, info):
        df = info.data.get("df")  # Access `df` from ValidationInfo
        if df is not None and target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' is not present in the DataFrame columns")
        return target_variable
