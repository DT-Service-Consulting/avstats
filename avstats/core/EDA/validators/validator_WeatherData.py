# core/EDA/validators/validator_WeatherData.py
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Any
import pandas as pd


class WeatherDataInput(BaseModel):
    df: pd.DataFrame
    weather_records: List[Any] = Field(default_factory=list)
    weather_df: Optional[pd.DataFrame] = None
    airports: dict = Field(default_factory=dict)
    custom_coords: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> None:
        """
        Validates the input DataFrame to ensure it contains the required columns.

        Args:
            df (pd.DataFrame): The input DataFrame to validate.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        required_columns = ['dep_iata_code', 'arr_iata_code', 'adt', 'aat']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    @staticmethod
    def validate_custom_coords(custom_coords: dict) -> None:
        """
        Validates the custom coordinates dictionary to ensure proper structure.

        Args:
            custom_coords (dict): The custom coordinates dictionary to validate.

        Raises:
            ValueError: If the dictionary has improperly formatted values.
        """
        for key, value in custom_coords.items():
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(
                    f"Custom coordinates for {key} must be a tuple of (latitude, longitude)."
                )
