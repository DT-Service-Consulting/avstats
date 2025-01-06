# core/ML_workflow/validators_ML/validator_time_series_analysis.py
from pydantic import BaseModel, field_validator, ConfigDict
import pandas as pd
from datetime import datetime


class TimeSeriesAnalysisInput(BaseModel):
    df: pd.DataFrame
    start_date: datetime
    end_date: datetime
    train_end: datetime
    test_end: datetime
    column: str
    date_column: str

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Set arbitrary_types_allowed

    @field_validator("df", mode="before")
    def validate_dataframe(cls, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        return df

    @field_validator("start_date", "end_date", "train_end", "test_end", mode="before")
    def validate_dates(cls, value):
        if not isinstance(value, datetime):
            raise ValueError("Dates must be valid datetime objects")
        return value

    @field_validator("column", "date_column")
    def validate_columns(cls, column, values):
        df = values.get("df")
        if df is not None and column not in df.columns:
            raise ValueError(f"Column '{column}' is not present in the DataFrame columns")
        return column

    @field_validator("end_date")
    def validate_date_range(cls, end_date, values):
        start_date = values.get("start_date")
        if start_date and end_date <= start_date:
            raise ValueError("end_date must be after start_date")
        return end_date