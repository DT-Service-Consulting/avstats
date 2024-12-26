# PassengerData.py
import pandas as pd


class PassengerData:
    def __init__(self, df, airport_mapping):
        self.df = df.copy()  # Make a copy of the DataFrame to avoid modifying the original
        self.airport_mapping = airport_mapping

    def convert_to_route_code(self, airport_entry):
        """
        Helper function for converting airport entries to route codes
        """
        airports = airport_entry.split(' - ')
        if len(airports) == 2:
            return f"{self.airport_mapping.get(airports[0].strip(), '')}-{self.airport_mapping.get(airports[1].strip(), '')}"
        return None

    def process_passenger_data(self):
        """
        Process a DataFrame of passenger airline data by cleaning and transforming it.

        Returns:
            pd.DataFrame: The processed DataFrame.
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
