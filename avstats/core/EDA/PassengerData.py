# PassengerData.py
import pandas as pd
from typing import Optional, Dict
from avstats.core.EDA.validators.validator_PassengerData import PassengerDataInput


class PassengerData:
    def __init__(self, df: pd.DataFrame, airport_mapping: Dict[str, str]):
        """
        Initialize the PassengerData class.

        Args:
            df (pd.DataFrame): The DataFrame containing passenger airline data.
            airport_mapping (Dict[str, str]): A mapping of airport codes to route codes.
        """
        self.df = df.copy()  # Make a copy of the DataFrame to avoid modifying the original
        self.airport_mapping = airport_mapping

    def convert_to_route_code(self, airport_entry: str) -> Optional[str]:
        """
        Convert an airport entry to a route code.

        Args:
            airport_entry (str): A string representing the airport entry in the format 'Airport1 - Airport2'.

        Returns:
            Optional[str]: The route code in the format 'Code1-Code2' if both airports are valid,
                           None otherwise.
        """
        airports = airport_entry.split(' - ')
        if len(airports) == 2:
            return f"{self.airport_mapping.get(airports[0].strip(), '')}-{self.airport_mapping.get(airports[1].strip(), '')}"
        return None

    def process_passenger_data(self) -> pd.DataFrame:
        """
        Process the passenger airline data by cleaning and transforming it.

        - Drops rows with any NaN values.
        - Removes rows where any column contains a ":" character.
        - Renames the 'TIME' column to 'AIRP_PR'.
        - Converts 'AIRP_PR' column entries to route codes and inserts them as a new column.
        - Drops the 'AIRP_PR' column.

        Returns:
            pd.DataFrame: The cleaned and transformed DataFrame.
        """
        # Drop rows with any NaN values
        self.df = self.df.dropna()

        # Drop rows where any column contains ":"
        mask = ~self.df.apply(lambda row: row.astype(str).str.contains(":").any(), axis=1)
        self.df = self.df.loc[mask]

        # Rename columns
        self.df.rename(columns={'TIME': 'AIRP_PR'}, inplace=True)

        # Apply the function to the 'AIRP_PR' column and create 'Route_Code' column
        self.df['route_code'] = self.df['AIRP_PR'].apply(self.convert_to_route_code)

        # Insert 'Route_Code' column right after 'AIRP_PR'
        route_code_col = self.df.pop('route_code')
        self.df.insert(self.df.columns.get_loc('AIRP_PR') + 1, 'route_code', route_code_col)

        # Remove the 'AIRP_PR' column
        self.df = self.df.drop(columns=['AIRP_PR'])

        return self.df
