# WeatherData.py
from meteostat import Daily, Point
import pandas as pd
from datetime import datetime

class WeatherData:
    def __init__(self, df):
        self.df = df
        self.weather_records = []
        self.weather_df = None
        self.airports = {}  # Airports dictionary is initialized for test

    def get_coordinates(self, iata_code: str):
        """
        Retrieve latitude and longitude for a given IATA code.

        Args:
            iata_code (str): IATA code for the airport.

        Returns:
            tuple: Latitude and longitude of the airport, or (None, None) if not found.
        """
        airport_info = self.airports.get(iata_code)
        if airport_info:
            return airport_info['lat'], airport_info['lon']
        return None, None

    def assign_coordinates(self):
        """
        Assigns latitude and longitude to departure and arrival airports in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing 'dep_iata_code' and 'arr_iata_code'.

        Returns:
            pd.DataFrame: Updated DataFrame with latitude and longitude for departure and arrival.
        """
        print("Assigning coordinates...")

        # Create latitude and longitude columns for both departure and arrival
        self.df[['dep_lat', 'dep_lon']] = self.df['dep_iata_code'].apply(
            lambda x: pd.Series(self.get_coordinates(x)))
        self.df[['arr_lat', 'arr_lon']] = self.df['arr_iata_code'].apply(
            lambda x: pd.Series(self.get_coordinates(x)))

        # Check for missing values and print a message
        missing_dep_coords = self.df[self.df['dep_lat'].isnull() | self.df['dep_lon'].isnull()]
        missing_arr_coords = self.df[self.df['arr_lat'].isnull() | self.df['arr_lon'].isnull()]

        print("DataFrame after assigning coordinates:", self.df.head())
        if not missing_dep_coords.empty:
            print(
                f"Missing coordinates for departure airports in the following routes:\n{missing_dep_coords[['dep_iata_code']]}")

        if not missing_arr_coords.empty:
            print(
                f"Missing coordinates for arrival airports in the following routes:\n{missing_arr_coords[['arr_iata_code']]}")

        return self.df

    def fetch_weather_data(self):
        # Extract unique dates for both departure and arrival
        unique_dates = pd.to_datetime(self.df[['adt', 'aat']].stack().unique()).date
        unique_dates = [datetime(d.year, d.month, d.day) for d in unique_dates]

        start_date, end_date = min(unique_dates), max(unique_dates)

        # Extract unique departure and arrival coordinates
        unique_dep_coords = self.df[['dep_lat', 'dep_lon', 'dep_iata_code']].drop_duplicates()
        unique_arr_coords = self.df[['arr_lat', 'arr_lon', 'arr_iata_code']].drop_duplicates()
        all_coords = pd.concat([unique_dep_coords, unique_arr_coords]).drop_duplicates()

        print(f"Starting weather data fetch for {len(all_coords)} coordinate pairs from {start_date} to {end_date}.")

        # Iterate over unique coordinates, fetching for the full date range
        for i, row in all_coords.iterrows():
            lat, lon, iata_code = row[['dep_lat', 'dep_lon', 'dep_iata_code']]
            # Fetch weather data without date parsing
            point = Point(lat, lon)
            weather_data = Daily(point, start_date, end_date).fetch()

            if not weather_data.empty:
                weather_data.reset_index(inplace=True)  # Ensure 'time' or 'date' is a column for merging
                # Explicitly convert 'time' or 'date' column to datetime after fetching
                if 'time' in weather_data.columns:
                    weather_data['time'] = pd.to_datetime(weather_data['time'])
                elif 'date' in weather_data.columns:
                    weather_data['date'] = pd.to_datetime(weather_data['date'])

                weather_data['lat'], weather_data['lon'], weather_data['iata_code'] = lat, lon, iata_code
                self.weather_records.append(weather_data)

            if (i + 1) % 100 == 0:
                print(f"Fetched weather for {i + 1} / {len(all_coords)} coordinates.")

        self.weather_df = pd.concat(self.weather_records, ignore_index=True) if self.weather_records else pd.DataFrame()
        print("Weather data fetching completed.")

    def merge_weather_with_flights(self) -> pd.DataFrame:
        """
        Merges the flight data with the corresponding weather data for both
        departure and arrival cities.

        Returns:
            pd.DataFrame: Merged DataFrame containing flight and weather information.
        """
        # Ensure 'time' column in weather_df is in datetime format for merging
        self.weather_df['time'] = self.weather_df['time'].dt.date

        # Ensure that 'adt' and 'aat' columns in df are also date-only for merging
        self.df['adt_date'] = pd.to_datetime(self.df['adt']).dt.date
        self.df['aat_date'] = pd.to_datetime(self.df['aat']).dt.date

        # Merge departure city weather data
        dep_weather = self.weather_df.rename(columns={
            'time': 'adt_date',
            'iata_code': 'dep_iata_code',
            'tavg': 'tavg_dep', 'tmin': 'tmin_dep', 'tmax': 'tmax_dep', 'prcp': 'prcp_dep',
            'snow': 'snow_dep', 'wdir': 'wdir_dep', 'wspd': 'wspd_dep', 'wpgt': 'wpgt_dep',
            'pres': 'pres_dep', 'tsun': 'tsun_dep', 'lat': 'dep_lat', 'lon': 'dep_lon'
        })

        # Merge arrival city weather data
        arr_weather = self.weather_df.rename(columns={
            'time': 'aat_date',
            'iata_code': 'arr_iata_code',
            'tavg': 'tavg_arr', 'tmin': 'tmin_arr', 'tmax': 'tmax_arr', 'prcp': 'prcp_arr',
            'snow': 'snow_arr', 'wdir': 'wdir_arr', 'wspd': 'wspd_arr', 'wpgt': 'wpgt_arr',
            'pres': 'pres_arr', 'tsun': 'tsun_arr', 'lat': 'arr_lat', 'lon': 'arr_lon'
        })

        # Merge with arrival weather data on arr_iata_code and aat_date
        self.df = pd.merge(self.df, dep_weather, on=['dep_iata_code', 'adt_date'], how='left')
        self.df = pd.merge(self.df, arr_weather, on=['arr_iata_code', 'aat_date'], how='left')

        # Drop auxiliary date columns if no longer needed
        self.df.drop(columns=['adt_date', 'aat_date'], inplace=True)

        print("Weather data was merged with schedule data successfully.")
        return self.df