# core/ML/validators/validator_OneHotEncoding.py
from pydantic import BaseModel, ConfigDict, field_validator
import pandas as pd


class OneHotEncodingInput(BaseModel):
    df: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df")
    def validate_dataframe_columns(cls, df: pd.DataFrame):
        # Ensure required columns exist
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
        supported_types = {"object", "int64", "float64", "datetime64[ns]"}

        # Identify unsupported columns
        unsupported_columns = [
            col for col, dtype in df.dtypes.items() if dtype.name not in supported_types
        ]
        if unsupported_columns:
            raise ValueError(f"Dataframe contains unsupported data types in columns: {unsupported_columns}")

        # Handle null values: either drop or fill them
        null_columns = df.columns[df.isnull().any()].tolist()
        if null_columns:
            # Optionally, you can choose to fill NaN values instead of raising an error
            df = df.fillna(0)  # Fill NaN with 0, or apply a different strategy
            # raise ValueError(f"Dataframe contains null values in columns: {null_columns}")

        return df
