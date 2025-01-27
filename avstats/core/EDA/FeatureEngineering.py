# FeatureEngineering.py
import pandas as pd
from avstats.core.EDA.validators.validator_NewFeatures import CategorizeFlightInput, GetTimeWindowInput


class FeatureEngineering:
    def __init__(self, df) -> None:
        """
        Initialize the DataCleaning class with a unique column.

        Parameters:
        unique_column (str): The name of the column to check for duplicates.
        """
        self.df = df

    @staticmethod
    def categorize_flight(cargo: bool, private: bool) -> str:
        """
        Categorize the flight type as 'Cargo', 'Private', or 'Commercial'.

        Returns:
        str: The category of the flight.
        """
        # Validate inputs using Pydantic
        validated_input = CategorizeFlightInput(cargo=cargo, private=private)

        if validated_input.cargo:
            return 'Cargo'
        elif validated_input.private:
            return 'Private'
        return 'Commercial'

    @staticmethod
    def get_time_window(hour: int) -> str:
        """
        Determine the time window (Morning, Afternoon, Evening) based on hour.

        Parameters:
        hour (int): Hour of the day.

        Returns:
        str: The time window.

        Raises:
        ValueError: If the hour is not between 0 and 23.
        """
        # Validate inputs using Pydantic
        validated_input = GetTimeWindowInput(hour=hour)

        if validated_input.hour < 0 or validated_input.hour > 23:
            raise ValueError("Hour must be between 0 and 23")
        if validated_input.hour < 12:
            return 'Morning'
        elif validated_input.hour < 18:
            return 'Afternoon'
        return 'Evening'

    def new_features(self) -> pd.DataFrame:
        # Feature engineering
        self.df['dep_delay_15'] = (self.df['dep_delay'] > 15).astype(int)
        self.df['dep_delay_cat'] = pd.cut(
            self.df['dep_delay'], bins=[-float('inf'), 15, 60, float('inf')], labels=['Short', 'Medium', 'Long'])

        self.df['flight_cat'] = self.df.apply(
            lambda row: self.categorize_flight(row['cargo'], row['private']), axis=1)

        self.df['dep_time_window'] = self.df['adt'].apply(
            lambda x: self.get_time_window(x.hour) if pd.notnull(x) else None)

        self.df['arr_time_window'] = self.df['aat'].apply(
            lambda x: self.get_time_window(x.hour) if pd.notnull(x) else None)

        self.df['on_time_15'] = (self.df['dep_delay'] < 15).astype(int)
        return self.df
