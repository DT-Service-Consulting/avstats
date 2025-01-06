# core/ML_workflow/validators_ML/validator_OneHotEncoding.py
from pydantic import BaseModel, ConfigDict, field_validator, ValidationError
import pandas as pd


class OneHotEncodingInput(BaseModel):
    df: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df")
    def validate_dataframe_columns(cls, df: pd.DataFrame):
        required_columns = {'route_iata_code', 'total_dep_delay_15'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return df

    @field_validator("df", mode="after")
    def ensure_valid_data(cls, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Input dataframe is empty.")
        if not all(df.dtypes.apply(lambda dtype: dtype.name in ["object", "int64", "float64"])):
            raise ValueError("Dataframe contains unsupported data types.")
        return df
