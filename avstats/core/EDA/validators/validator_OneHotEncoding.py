# core/ML/validators/validator_OneHotEncoding.py
from pydantic import BaseModel, ConfigDict, field_validator
import pandas as pd


class OneHotEncodingInput(BaseModel):
    df: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df", mode="after")
    def validate_dataframe_columns(cls, df: pd.DataFrame):
        """
        Ensure required columns exist and validate optional interchangeable columns.
        """
        required_column = {'route_iata_code'}
        optional_column = {'dep_delay_15', 'total_dep_delay_15'}

        # Check for the required column
        missing_columns = required_column - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check that at least one of the interchangeable columns exists
        if not optional_column.intersection(set(df.columns)):
            raise ValueError("Missing required column: 'dep_delay_15' or 'total_dep_delay_15'")

        return df

    @field_validator("df", mode="after")
    def ensure_valid_data(cls, df: pd.DataFrame):
        """
        Validate data types and handle nulls in the dataframe.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        # Supported types (updated to include specific types if necessary)
        supported_types = {"object", "int64", "float64", "datetime64[ns]", "bool"}

        # Identify unsupported columns
        unsupported_columns = [
            col for col, dtype in df.dtypes.items() if dtype.name not in supported_types
        ]
        if unsupported_columns:
            # Try to coerce unsupported columns to object
            for col in unsupported_columns:
                try:
                    df[col] = df[col].astype('object')
                except Exception:
                    raise ValueError(f"Dataframe contains unsupported data types in columns: {unsupported_columns}")

        # Handle null values: either drop or fill them
        null_columns = df.columns[df.isnull().any()].tolist()
        if null_columns:
            df = df.fillna(0)  # Fill NaN with 0, or apply a different strategy

        return df
