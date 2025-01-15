# FlightPerformance.py
import pandas as pd
from typing import List, Tuple


class FlightPerformance:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the FlightPerformance class with a DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame containing flight data.
        """
        self.df = dataframe

    def overall_performance(self) -> dict:
        """
        Calculate overall flight performance.

        Returns:
        dict: Overall performance metrics.
        """
        total_flights = len(self.df)
        if total_flights == 0:
            return {"Delayed Flights (%)": 0, "On-Time Flights (%)": 0, "Missing Status (%)": 0,}

        delayed = self.df['dep_delay_15'].sum()
        on_time = self.df['on_time_15'].sum()
        missing = total_flights - (delayed + on_time)

        return {
            "Delayed Flights (%)": (delayed / total_flights) * 100,
            "On-Time Flights (%)": (on_time / total_flights) * 100,
            "Missing Status (%)": (missing / total_flights) * 100,
        }

    def delay_ranges_summary(self, delay_ranges: List[Tuple[int, int]]) -> dict:
        """
        Calculate delayed flight percentages within specified delay ranges.

        Parameters:
        delay_ranges (List[Tuple[int, int]]): List of (min_delay, max_delay) pairs.

        Returns:
        dict: Delay range summaries.
        """
        total_flights = len(self.df)
        if total_flights == 0:
            return {f"{lower}-{upper} minutes": 0 for lower, upper in delay_ranges}

        return {
            f"{lower}-{upper} minutes": ((self.df['dep_delay'] > lower) & (
                        self.df['dep_delay'] <= upper)).sum() / total_flights * 100
            for lower, upper in delay_ranges
        }
