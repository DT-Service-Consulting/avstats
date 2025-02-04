# FeatureEngineering.py
import pandas as pd
from avstats.core.EDA.validators.validator_FeatureEngineering import CategorizeFlightInput, GetTimeWindowInput


class FeatureEngineering:
    def __init__(self, df) -> None:
        """
        Initialize the FeatureEngineering class.

        Parameters:
        df (pd.DataFrame): The input dataframe.
        """
        self.df = df

    @staticmethod
    def categorize_flight(cargo: bool, private: bool) -> str:
        """
        Categorize the flight type as 'Cargo', 'Private', or 'Commercial'.

        Returns:
        str: The category of the flight.
        """
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

        Returns:
        str: The time window.
        """
        validated_input = GetTimeWindowInput(hour=hour)
        if validated_input.hour < 12:
            return 'Morning'
        elif validated_input.hour < 18:
            return 'Afternoon'
        return 'Evening'

    @staticmethod
    def get_season(month: int) -> str:
        """
        Determine the season based on the month.

        Returns:
        str: The season.
        """
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        return 'Autumn'

    @staticmethod
    def new_weather_features(df) -> pd.DataFrame:
        """
        Generate new weather-related features for the dataset with absolute values.

        Returns:
        pd.DataFrame: Dataframe with additional features.
        """
        # Weather-related features (Ensure columns exist)
        if {'tavg_dep', 'tavg_arr'}.issubset(df.columns):
            df['temp_diff'] = (df['tavg_dep'] - df['tavg_arr']).abs()
        if {'wspd_dep', 'wspd_arr'}.issubset(df.columns):
            df['wind_speed_diff'] = (df['wspd_dep'] - df['wspd_arr']).abs()
        return df

    def new_features(self) -> pd.DataFrame:
        """
        Generate new features for the dataset.

        Returns:
        pd.DataFrame: Dataframe with additional features.
        """
        # Delay categories
        self.df['dep_delay_15'] = (self.df['dep_delay'] > 15).astype(int)
        self.df['dep_delay_cat'] = pd.cut(
            self.df['dep_delay'], bins=[-float('inf'), 15, 60, float('inf')], labels=['Short', 'Medium', 'Long'])

        # Flight category
        self.df['flight_cat'] = self.df.apply(
            lambda row: self.categorize_flight(row['cargo'], row['private']), axis=1)

        # Time-based features
        self.df['dep_time_window'] = self.df['adt'].apply(
            lambda x: self.get_time_window(x.hour) if pd.notnull(x) else None)
        self.df['arr_time_window'] = self.df['aat'].apply(
            lambda x: self.get_time_window(x.hour) if pd.notnull(x) else None)

        self.df['day_of_week'] = self.df['sdt'].dt.dayofweek
        self.df['month'] = self.df['sdt'].dt.month
        self.df['hour_of_day'] = self.df['sdt'].dt.hour
        self.df['season'] = self.df['month'].apply(self.get_season)

        # Time-related binary feature
        self.df['on_time_15'] = (self.df['dep_delay'] < 15).astype(int)

        # Historical airline delays (requires precomputed mean delays)
        if 'airline_iata_code' in self.df.columns:
            airline_avg_delays = self.df.groupby('airline_iata_code')['dep_delay'].mean()
            self.df['historical_airline_delays'] = self.df['airline_iata_code'].map(airline_avg_delays)

        # Route-specific delays
        if 'route_iata_code' in self.df.columns:
            route_avg_delays = self.df.groupby('route_iata_code')['dep_delay'].mean()
            self.df['route_delays'] = self.df['route_iata_code'].map(route_avg_delays)

        return self.df
