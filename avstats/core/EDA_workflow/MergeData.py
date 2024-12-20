# MergeData.py
import pandas as pd
import numpy as np


class MergeData:
    def __init__(self, df):
        """
        Initialize the class with a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with flight and weather data.
        """
        self.df = df.copy()

    def preprocess_datetime(self, datetime_col):
        """
        Preprocess datetime column: ensure timezone-aware and convert to UTC.

        Args:
            datetime_col (str): Name of the datetime column to process.
        """
        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col], errors='coerce')
        if self.df[datetime_col].dt.tz is None:
            self.df[datetime_col] = self.df[datetime_col].dt.tz_localize('UTC')
        self.df[datetime_col] = self.df[datetime_col].dt.tz_convert('UTC').dt.tz_localize(None)
        self.df['Date'] = self.df[datetime_col].dt.date

    def aggregate_daily(self):
        """
        Perform daily aggregation on the data.

        Returns:
            pd.DataFrame: Aggregated daily data grouped by route and date.
        """
        aggregated_df = (
            self.df.groupby(['route_iata_code', 'Date'])
            .agg(
                total_flights=('uuid', 'count'),

                # Type counts
                departures=('type', lambda x: np.sum(x == 'DEPARTURE')),
                arrivals=('type', lambda x: np.sum(x == 'ARRIVAL')),

                # Status counts
                landed=('status', lambda x: np.sum(x == 'LANDED')),
                active=('status', lambda x: np.sum(x == 'ACTIVE')),
                scheduled=('status', lambda x: np.sum(x == 'SCHEDULED')),

                # Delays and on-time flights
                total_dep_delay=('dep_delay', 'sum'),
                total_dep_delay_15=('dep_delay_15', 'sum'),
                total_on_time_15=('on_time_15', 'sum'),

                # Delay categories
                short_delay=('dep_delay_cat', lambda x: np.sum(x == 'Short')),
                medium_delay=('dep_delay_cat', lambda x: np.sum(x == 'Medium')),
                long_delay=('dep_delay_cat', lambda x: np.sum(x == 'Long')),

                # Calculated metrics
                total_calc_sft=('calc_sft', 'sum'),
                total_calc_aft=('calc_aft', 'sum'),
                total_flight_distance_km=('calc_flight_distance_km', 'sum'),

                # Flight categories
                commercial=('flight_cat', lambda x: np.sum(x == 'Commercial')),
                private=('flight_cat', lambda x: np.sum(x == 'Private')),
                cargo=('flight_cat', lambda x: np.sum(x == 'Cargo')),

                # Departure time windows
                morning_dep=('dep_time_window', lambda x: np.sum(x == 'Morning')),
                afternoon_dep=('dep_time_window', lambda x: np.sum(x == 'Afternoon')),
                evening_dep=('dep_time_window', lambda x: np.sum(x == 'Evening')),

                # Arrival time windows
                morning_arr=('arr_time_window', lambda x: np.sum(x == 'Morning')),
                afternoon_arr=('arr_time_window', lambda x: np.sum(x == 'Afternoon')),
                evening_arr=('arr_time_window', lambda x: np.sum(x == 'Evening')),

                # Departure weather
                avg_tavg_dep=('tavg_dep', 'mean'),
                total_prcp_dep=('prcp_dep', 'sum'),
                avg_snow_dep=('snow_dep', 'mean'),
                avg_wdir_dep=('wdir_dep', 'mean'),
                avg_wspd_dep=('wspd_dep', 'mean'),
                avg_wpgt_dep=('wpgt_dep', 'mean'),
                avg_pres_dep=('pres_dep', 'mean'),
                total_tsun_dep=('tsun_dep', 'sum'),

                # Arrival weather
                avg_tavg_arr=('tavg_arr', 'mean'),
                total_prcp_arr=('prcp_arr', 'sum'),
                avg_snow_arr=('snow_arr', 'mean'),
                avg_wdir_arr=('wdir_arr', 'mean'),
                avg_wspd_arr=('wspd_arr', 'mean'),
                avg_wpgt_arr=('wpgt_arr', 'mean'),
                avg_pres_arr=('pres_arr', 'mean'),
                total_tsun_arr=('tsun_arr', 'sum'),
            )
            .reset_index()
        )
        return aggregated_df