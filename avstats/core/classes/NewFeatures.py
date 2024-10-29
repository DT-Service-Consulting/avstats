import pandas as pd


class NewFeatures:
    def __init__(self, uuid: str, dep_delay: int, sdt: pd.Timestamp, sat: pd.Timestamp, cargo: bool, private: bool):
        self.uuid = uuid
        self.dep_delay = dep_delay
        self.sdt = sdt
        self.sat = sat
        self.cargo = cargo
        self.private = private

    def categorize_flight(self) -> str:
        """
        Categorize the flight type as 'Cargo', 'Private', or 'Commercial'.

        Returns:
        str: The category of the flight.
        """
        if self.cargo:
            return 'Cargo'
        elif self.private:
            return 'Private'
        else:
            return 'Commercial'

    def get_time_window(self, time_type: str = 'departure') -> str:
        """
        Determine the time window of the flight (Morning, Afternoon, Evening).

        Parameters:
        time_type (str): Whether to calculate based on 'departure' or 'arrival'. Must be either 'departure' or 'arrival'.

        Returns:
        str: The time window ('Morning', 'Afternoon', or 'Evening').

        Raises:
        ValueError: If time_type is not 'departure' or 'arrival'.
        """
        if time_type == 'departure':
            hour = self.sdt.hour
        elif time_type == 'arrival':
            hour = self.sat.hour
        else:
            raise ValueError("time_type must be either 'departure' or 'arrival'.")

        if hour < 12:
            return 'Morning'
        elif hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'
