# core/ML/validators/validator_OneHotEncoding.py
from pydantic import BaseModel, ConfigDict, field_validator
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

        # Supported types
        supported_types = ["object", "int64", "float64", "datetime64[ns]"]

        # Identify unsupported columns
        unsupported_columns = df.dtypes[~df.dtypes.apply(lambda dtype: dtype.name in supported_types)].index.tolist()
        if unsupported_columns:
            raise ValueError(f"Dataframe contains unsupported data types in columns: {unsupported_columns}")

        # Check for excessive null values
        if df.isnull().any().any():
            null_columns = df.columns[df.isnull().any()].tolist()
            raise ValueError(f"Dataframe contains null values in columns: {null_columns}")

        return df
