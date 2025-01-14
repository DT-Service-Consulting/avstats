# MergeData.py
import pandas as pd
import numpy as np
from avstats.core.EDA.validators.validator_MergeData import MergeDataInput


class MergeData:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the class with a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with flight and weather data.
        """
        validated_input = MergeDataInput(df=df)  # Validate the input DataFrame
        self.df = validated_input.df.copy()
        self.df_grouped = None

    def preprocess_datetime(self) -> None:
        """
        Preprocess datetime column: ensure timezone-aware and convert to UTC.
        """
        self.df['sdt'] = pd.to_datetime(self.df['sdt'], errors='coerce')
        if self.df['sdt'].dt.tz is None:
            self.df['sdt'] = self.df['sdt'].dt.tz_localize('UTC')
        self.df['sdt'] = self.df['sdt'].dt.tz_convert('UTC').dt.tz_localize(None)

    def aggregate_daily(self, passenger_data: bool = True) -> pd.DataFrame:
        """
        Perform aggregation on the data grouped by either Month or Date.

        Args:
            passenger_data (bool): If True, group by 'Month'; otherwise, group by 'Date'.

        Returns:
            pd.DataFrame: Aggregated data grouped by route and the selected time period.
        """
        if passenger_data:
            self.df['Month'] = pd.to_datetime(self.df['sdt']).dt.to_period('M')
            group_by_col = 'Month'
            is_monthly = True
        else:
            self.df['Date'] = self.df['sdt'].dt.date
            group_by_col = 'Date'
            is_monthly = False

        # Define weather aggregation logic with dynamic column names
        weather_agg = {
             # Departure weather
            'tavg_dep': ('tavg_dep', 'mean' if is_monthly else 'first'),
            'prcp_dep': ('prcp_dep', 'sum' if is_monthly else 'first'),
            'snow_dep': ('snow_dep', 'mean' if is_monthly else 'first'),
            'wdir_dep': ('wdir_dep', 'mean' if is_monthly else 'first'),
            'wspd_dep': ('wspd_dep', 'mean' if is_monthly else 'first'),
            'wpgt_dep': ('wpgt_dep', 'mean' if is_monthly else 'first'),
            'pres_dep': ('pres_dep', 'mean' if is_monthly else 'first'),
            'tsun_dep': ('tsun_dep', 'sum' if is_monthly else 'first'),

            # Arrival weather
            'tavg_arr': ('tavg_arr', 'mean' if is_monthly else 'first'),
            'prcp_arr': ('prcp_arr', 'sum' if is_monthly else 'first'),
            'snow_arr': ('snow_arr', 'mean' if is_monthly else 'first'),
            'wdir_arr': ('wdir_arr', 'mean' if is_monthly else 'first'),
            'wspd_arr': ('wspd_arr', 'mean' if is_monthly else 'first'),
            'wpgt_arr': ('wpgt_arr', 'mean' if is_monthly else 'first'),
            'pres_arr': ('pres_arr', 'mean' if is_monthly else 'first'),
            'tsun_arr': ('tsun_arr', 'sum' if is_monthly else 'first'),
        }

        # Build the aggregation dictionary
        agg_dict = {
            'total_flights': ('uuid', 'count'),

            # Type counts
            'departures': ('type', lambda x: np.sum(x == 'DEPARTURE')),
            'arrivals': ('type', lambda x: np.sum(x == 'ARRIVAL')),

            # Status counts
            'landed': ('status', lambda x: np.sum(x == 'LANDED')),
            'active': ('status', lambda x: np.sum(x == 'ACTIVE')),
            'scheduled': ('status', lambda x: np.sum(x == 'SCHEDULED')),

            # Delays and on-time flights
            'total_dep_delay': ('dep_delay', 'sum'),
            'total_dep_delay_15': ('dep_delay_15', 'sum'),
            'total_on_time_15': ('on_time_15', 'sum'),

            # Delay categories
            'short_delay': ('dep_delay_cat', lambda x: np.sum(x == 'Short')),
            'medium_delay': ('dep_delay_cat', lambda x: np.sum(x == 'Medium')),
            'long_delay': ('dep_delay_cat', lambda x: np.sum(x == 'Long')),

            # Calculated metrics
            'total_calc_sft': ('calc_sft', 'sum'),
            'total_calc_aft': ('calc_aft', 'sum'),
            'total_flight_distance_km': ('calc_flight_distance_km', 'sum'),

            # Flight categories
            'commercial': ('flight_cat', lambda x: np.sum(x == 'Commercial')),
            'private': ('flight_cat', lambda x: np.sum(x == 'Private')),
            'cargo': ('flight_cat', lambda x: np.sum(x == 'Cargo')),

            # Departure time windows
            'morning_dep': ('dep_time_window', lambda x: np.sum(x == 'Morning')),
            'afternoon_dep': ('dep_time_window', lambda x: np.sum(x == 'Afternoon')),
            'evening_dep': ('dep_time_window', lambda x: np.sum(x == 'Evening')),

            # Arrival time windows
            'morning_arr': ('arr_time_window', lambda x: np.sum(x == 'Morning')),
            'afternoon_arr': ('arr_time_window', lambda x: np.sum(x == 'Afternoon')),
            'evening_arr': ('arr_time_window', lambda x: np.sum(x == 'Evening')),
        }

        # Update the aggregation dictionary with weather logic dynamically
        if is_monthly:
            # Rename columns for monthly aggregation
            weather_agg = {
                f"avg_{key}": (key, func) for key, (key, func) in weather_agg.items()
            }
        agg_dict.update(weather_agg)

        # Group and aggregate
        self.df_grouped = (
            self.df.groupby(['route_iata_code', group_by_col])
            .agg(**agg_dict)
            .reset_index()
        )

        return self.df_grouped

    def aggregate_passengers(self, df_passengers: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate passenger data by merging it with flight data.

        Args:
            df_passengers (pd.DataFrame): Passenger data with columns `route_code` and monthly passenger counts.

        Returns:
            pd.DataFrame: Merged DataFrame of aggregated flight and passenger data.
        """
        # Reshape to long format
        df_passengers_long = pd.melt(
            df_passengers,
            id_vars='route_code',
            var_name='Month',
            value_name='total_passengers'
        )
        # Convert Month to 'YYYY-MM' format
        df_passengers_long['Month'] = pd.to_datetime(df_passengers_long['Month'], format='%Y-%m').dt.to_period('M')

        df_merged = pd.merge(
            self.df_grouped,
            df_passengers_long,
            left_on=['route_iata_code', 'Month'],
            right_on=['route_code', 'Month'],
            how='inner'
        )

        # Drop the `route_code` column
        df_merged.drop(columns=['route_code'], inplace=True)

        return df_merged
