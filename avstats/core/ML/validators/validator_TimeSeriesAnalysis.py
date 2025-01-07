# core/ML/validators/validator_TimeSeriesAnalysis.py
from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo
import pandas as pd
from datetime import datetime


class TimeSeriesAnalysisInput(BaseModel):
    df: pd.DataFrame
    start_date: datetime
    end_date: datetime
    train_end: datetime
    test_end: datetime
    column: str

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Set arbitrary_types_allowed

    @field_validator("end_date")
    def validate_date_range(cls, end_date: datetime, info: ValidationInfo):
        start_date = info.data.get("start_date")  # Access start_date using info.data
        if start_date is not None and start_date >= end_date:
            raise ValueError("start_date must be earlier than end_date")
        return end_date

    @field_validator("train_end")
    def validate_train_end(cls, train_end: datetime, info: ValidationInfo):
        start_date = info.data.get("start_date")
        end_date = info.data.get("end_date")
        if start_date is not None and train_end < start_date:
            raise ValueError("train_end must be within the range of start_date and end_date")
        if end_date is not None and train_end >= end_date:
            raise ValueError("train_end must be earlier than end_date")
        return train_end

    @field_validator("test_end")
    def validate_test_end(cls, test_end: datetime, info: ValidationInfo):
        train_end = info.data.get("train_end")
        end_date = info.data.get("end_date")
        if train_end is not None and test_end <= train_end:
            raise ValueError("test_end must be later than train_end")
        if end_date is not None and test_end > end_date:
            raise ValueError("test_end must not exceed end_date")
        return test_end

    @field_validator("column")
    def validate_column(cls, column: str, info: ValidationInfo):
        df = info.data.get("df")
        if df is not None and column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the dataframe")
        return column
