# core/EDA/validators/validator_PassengerData.py
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Dict
import pandas as pd


class PassengerDataInput(BaseModel):
    df: pd.DataFrame
    airport_mapping: Dict[str, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df", mode="before")
    def validate_dataframe(cls, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")
        required_columns = {'TIME'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        return df

    @field_validator("airport_mapping", mode="before")
    def validate_airport_mapping(cls, airport_mapping):
        if not isinstance(airport_mapping, dict):
            raise ValueError("Input 'airport_mapping' must be a dictionary.")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in airport_mapping.items()):
            raise ValueError("All keys and values in 'airport_mapping' must be strings.")
        return airport_mapping
