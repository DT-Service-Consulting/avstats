# WeatherData.py
import pandas as pd
import warnings
import time
from meteostat import Daily, Point
from datetime import datetime
from airportsdata import load # (pip install airportsdata)
from typing import Optional
from avstats.core.EDA.validators.validator_WeatherData import WeatherDataInput
warnings.filterwarnings("ignore", category=FutureWarning)


class WeatherData:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the WeatherData class with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing flight schedule information.
        """
        # Validate inputs using the WeatherDataInput Pydantic model
        validated_inputs = WeatherDataInput(
            df=df,
            weather_records=[],
            weather_df=None,
            airports=load('IATA'),
            custom_coords={
                'KIV': (46.9279, 28.9303),  # Chișinău International Airport
                'CQM': (30.1225, 31.4041),  # Cluj International Airport
                'SZY': (53.4841, 20.7465),  # Olsztyn-Mazury Airport
            },
            missing_weather_records=[]
        )

        # Store validated inputs
        self.df = validated_inputs.df
        self.weather_records = validated_inputs.weather_records
        self.weather_df = validated_inputs.weather_df
        self.airports = validated_inputs.airports
        self.custom_coords = validated_inputs.custom_coords
        self.missing_weather_records = validated_inputs.missing_weather_records

    def get_coordinates(self, iata_code: str) -> tuple[Optional[float], Optional[float]]:
        """
        Retrieve latitude and longitude for a given IATA code.

        Args:
            iata_code (str): IATA code for the airport.

        Returns:
            tuple: Latitude and longitude of the airport, or (None, None) if not found.
        """
        # Check custom coordinates first
        if iata_code in self.custom_coords:
            return self.custom_coords[iata_code]

        # Fall back to airportsdata library
        airport_info = self.airports.get(iata_code)
        if airport_info:
            return airport_info['lat'], airport_info['lon']
        return None, None

    def assign_coordinates(self) -> pd.DataFrame:
        """
        Assign latitude and longitude to departure and arrival airports in the DataFrame.

        Returns:
            pd.DataFrame: Updated DataFrame with latitude and longitude for departure and arrival.
        """
        print("Assigning coordinates...")

        # Vectorized approach using map
        dep_coords = self.df['dep_iata_code'].map(lambda x: self.get_coordinates(x))
        arr_coords = self.df['arr_iata_code'].map(lambda x: self.get_coordinates(x))

        # Split tuples into separate latitude and longitude columns
        self.df['dep_lat'], self.df['dep_lon'] = zip(*dep_coords)
        self.df['arr_lat'], self.df['arr_lon'] = zip(*arr_coords)

        # Check for missing values
        missing_dep_coords = self.df[self.df['dep_lat'].isnull() | self.df['dep_lon'].isnull()]
        missing_arr_coords = self.df[self.df['arr_lat'].isnull() | self.df['arr_lon'].isnull()]

        if not missing_dep_coords.empty:
            print(
                f"Missing coordinates for departure airports in the following routes:\n{missing_dep_coords[['dep_iata_code']]}")

        if not missing_arr_coords.empty:
            print(
                f"Missing coordinates for arrival airports in the following routes:\n{missing_arr_coords[['arr_iata_code']]}")

        return self.df

    def fetch_weather_data(self) -> None:
        """
        Fetch weather data for unique departure and arrival coordinates within the schedule's date range.

        This method populates the `weather_records` list with weather data and logs missing weather data.
        """
        # Extract unique dates for both departure and arrival
        unique_dates = pd.to_datetime(self.df[['adt', 'aat']].stack().unique())
        unique_dates = [datetime(d.year, d.month, d.day) for d in unique_dates if pd.notnull(d)]
        start_date, end_date = min(unique_dates), max(unique_dates)

        # Extract unique departure and arrival coordinates
        unique_dep_coords = self.df[['dep_lat', 'dep_lon', 'dep_iata_code']].drop_duplicates()
        unique_arr_coords = self.df[['arr_lat', 'arr_lon', 'arr_iata_code']].drop_duplicates()

        # Add a column to identify if the row is a departure or arrival coordinate
        unique_dep_coords['type'] = 'dep'
        unique_arr_coords['type'] = 'arr'

        # Rename columns for merging purposes
        unique_dep_coords.columns = ['lat', 'lon', 'iata_code', 'type']
        unique_arr_coords.columns = ['lat', 'lon', 'iata_code', 'type']

        # Concatenate and drop duplicates
        all_coords = pd.concat([unique_dep_coords, unique_arr_coords]).drop_duplicates()
        print(f"Starting weather data fetch for {len(all_coords)} coordinate pairs from {start_date} to {end_date}.")

        # List to store failed fetches
        missing_weather_records = []

        # Iterate over unique coordinates, fetching weather data
        for i, row in all_coords.iterrows():
            lat, lon, iata_code = row[['lat', 'lon', 'iata_code']]
            point = Point(lat, lon)
            retries = 3  # Number of retries for each fetch

            for attempt in range(retries):
                try:
                    weather_data = Daily(point, start_date, end_date).fetch()
                    if not weather_data.empty:
                        weather_data = weather_data.reset_index(drop=False)
                        weather_data['lat'], weather_data['lon'], weather_data['iata_code'] = lat, lon, iata_code
                        self.weather_records.append(weather_data)
                    break

                except Exception as e:
                    if attempt == retries - 1:
                        missing_weather_records.append({'lat': lat, 'lon': lon, 'iata_code': iata_code})
                    time.sleep(1)  # Wait before retrying

            # Progress update
            if (i + 1) % 100 == 0:
                print(f"Fetched weather for {i + 1} / {len(all_coords)} coordinates.")

            # Optional: Sleep to avoid API throttling
            time.sleep(0.1)  # Adjust as needed for API limits

        # Combine all fetched weather data into a single DataFrame
        self.weather_df = pd.concat(self.weather_records, ignore_index=True) if self.weather_records else pd.DataFrame()
        print(f"Weather data fetching completed with {len(self.weather_df)} records.")

        # Save missing weather data to a file for user inspection
        if missing_weather_records:
            missing_weather_df = pd.DataFrame(missing_weather_records)
            missing_weather_file = "missing_weather_data.csv"
            missing_weather_df.to_csv(missing_weather_file, index=False)
            print(f"Weather data could not be fetched for {len(missing_weather_records)} routes. "
                  f"Details saved to '{missing_weather_file}'.")

    def merge_weather_with_flights(self) -> pd.DataFrame:
        """
        Merge the flight data with the corresponding weather data for both
        departure and arrival cities.

        Returns:
            pd.DataFrame: Merged DataFrame containing flight and weather information.
        """
        # Ensure 'adt' and 'aat' columns are in datetime format
        self.df['adt'] = pd.to_datetime(self.df['adt'], errors='coerce')
        self.df['aat'] = pd.to_datetime(self.df['aat'], errors='coerce')

        # Ensure 'time' column in weather_df is in datetime format for merging
        self.weather_df['time'] = pd.to_datetime(self.weather_df['time']).dt.date
        self.df['adt_date'] = self.df['adt'].dt.date
        self.df['aat_date'] = self.df['aat'].dt.date
        print("adt and aat were successfully turned into datetime format.")

        # Merge departure city weather data
        dep_weather = self.weather_df.rename(columns={
            'time': 'adt_date', 'iata_code': 'dep_iata_code',
            'tavg': 'tavg_dep', 'tmin': 'tmin_dep', 'tmax': 'tmax_dep', 'prcp': 'prcp_dep',
            'snow': 'snow_dep', 'wdir': 'wdir_dep', 'wspd': 'wspd_dep', 'wpgt': 'wpgt_dep',
            'pres': 'pres_dep', 'tsun': 'tsun_dep', 'lat': 'dep_lat', 'lon': 'dep_lon'
        })

        # Merge arrival city weather data
        arr_weather = self.weather_df.rename(columns={
            'time': 'aat_date', 'iata_code': 'arr_iata_code',
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
