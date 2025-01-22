# core/EDA/validators/validator_WeatherData.py
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional, Any
import pandas as pd


class WeatherDataInput(BaseModel):
    df: pd.DataFrame
    weather_records: List[Any] = Field(default_factory=list)
    weather_df: Optional[pd.DataFrame] = None
    airports: dict = Field(default_factory=dict)
    custom_coords: dict = Field(default_factory=dict)
    missing_weather_records: List[dict] = Field(default_factory=list)  # New field for tracking missing weather data

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

    @field_validator("df")
    def validate_input_dataframe(cls, value: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the DataFrame provided as input.

        Args:
            value (pd.DataFrame): The input DataFrame to validate.

        Returns:
            pd.DataFrame: The validated DataFrame.

        Raises:
            ValueError: If required columns are missing or if the DataFrame is empty.
        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if value.empty:
            raise ValueError("The input DataFrame cannot be empty.")
        cls.validate_dataframe(value)  # Validate required columns
        return value

    @field_validator("weather_records")
    def validate_weather_records(cls, value: List[Any]) -> List[Any]:
        """
        Validates the weather_records list.

        Args:
            value (List[Any]): The weather records.

        Returns:
            List[Any]: The validated list.
        """
        if not isinstance(value, list):
            raise ValueError("weather_records must be a list.")
        return value

    @field_validator("weather_df")
    def validate_weather_df(cls, value: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Validates the weather DataFrame.

        Args:
            value (Optional[pd.DataFrame]): The weather DataFrame.

        Returns:
            Optional[pd.DataFrame]: The validated DataFrame.

        Raises:
            ValueError: If the DataFrame contains invalid data.
        """
        if value is not None:
            if not isinstance(value, pd.DataFrame):
                raise ValueError("weather_df must be a pandas DataFrame.")
            if value.empty:
                raise ValueError("weather_df cannot be empty if provided.")
        return value

    @field_validator("airports")
    def validate_airports(cls, value: dict) -> dict:
        """
        Validates the airports dictionary.

        Args:
            value (dict): The airports dictionary.

        Returns:
            dict: The validated dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("airports must be a dictionary.")
        return value

    @field_validator("custom_coords")
    def validate_custom_coords(cls, value: dict) -> dict:
        """
        Validates the custom_coords dictionary.

        Args:
            value (dict): The custom coordinates dictionary.

        Returns:
            dict: The validated dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("custom_coords must be a dictionary.")
        return value

    @field_validator("missing_weather_records")
    def validate_missing_weather_records(cls, value: List[dict]) -> List[dict]:
        """
        Validates the missing_weather_records list.

        Args:
            value (List[dict]): The list of missing weather records.

        Returns:
            List[dict]: The validated list.

        Raises:
            ValueError: If any record in the list is not a dictionary.
        """
        if not isinstance(value, list):
            raise ValueError("missing_weather_records must be a list.")
        for record in value:
            if not isinstance(record, dict):
                raise ValueError("Each entry in missing_weather_records must be a dictionary.")
        return value
