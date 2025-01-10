# core/EDA/validators/validator_DataProcessing.py
from pydantic import BaseModel, ConfigDict, field_validator
import pandas as pd


class DataProcessingInput(BaseModel):
    df: pd.DataFrame
    unique_column: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("unique_column")
    def validate_unique_column(cls, unique_column: str, values) -> str:
        # Access df from values.data
        df = values.data.get("df")
        if df is not None and unique_column not in df.columns:
            raise ValueError(f"Column '{unique_column}' does not exist in the DataFrame.")
        return unique_column
