import pandas as pd
from typing import List, Tuple, Dict
from avstats.core.general_utils import calc_percentage, count_delayed_flights


class FlightPerformance:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the FlightPerformance class with a DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame containing flight data.
        """
        self.df = dataframe

    def overall_performance(self) -> Dict[str, float]:
        """
        Calculate the overall performance of flights.

        Returns:
        dict: A dictionary containing the percentage of delayed flights,
              on-time flights, and flights with missing status.
        """
        total_flights = len(self.df)
        delayed_flights = self.df['dep_delay_15'].sum()
        ontime_flights = self.df['on_time_15'].sum()
        missing_status_flights = total_flights - (delayed_flights + ontime_flights)

        return {
            "Delayed Flights": calc_percentage(delayed_flights, total_flights),
            "On-Time Flights": calc_percentage(ontime_flights, total_flights),
            "Flights with Missing Status": calc_percentage(missing_status_flights, total_flights)
        }

    def delayed_flight_percentages(self, delay_ranges: List[Tuple[int, int, str]]) -> Dict[str, float]:
        """
        Calculate the percentage of delayed flights within specified delay ranges.

        Parameters:
        delay_ranges (List[Tuple[int, int, str]]): A list of tuples where each tuple contains the lower bound,
                                                   upper bound, and label for the delay range.

        Returns:
        dict: A dictionary with delay range labels as keys and their corresponding percentages as values.
        """
        total_flights = len(self.df)

        return {
            label: calc_percentage(
                count_delayed_flights(self.df, lower, upper),
                total_flights
            )
            for lower, upper, label in delay_ranges
        }
