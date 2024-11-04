# test_weather_data.py
import pytest
from pandas import DataFrame
from unittest.mock import patch
from datetime import datetime
from avstats.core.classes.WeatherData import WeatherData


@pytest.fixture
def sample_flight_data():
    data = {
        'dep_iata_code': ['JFK', 'LHR'],
        'arr_iata_code': ['SFO', 'DXB'],
        'adt': ['2023-01-01 06:00', '2023-01-02 08:00'],
        'aat': ['2023-01-01 12:00', '2023-01-02 22:00']
    }
    return DataFrame(data)


@pytest.fixture
def weather_data_class(sample_flight_data):
    # Add a dummy airports dictionary
    airports = {
        'JFK': {'lat': 40.6413, 'lon': -73.7781},
        'LHR': {'lat': 51.4700, 'lon': -0.4543},
        'SFO': {'lat': 37.6188056, 'lon': -122.3754167},
        'DXB': {'lat': 25.2532, 'lon': 55.3657}
    }
    weather_data = WeatherData(sample_flight_data)
    weather_data.airports = airports  # Assign the dummy airport data

    return weather_data


@pytest.fixture
def mock_fetch():
    with patch("meteostat.Daily.fetch", return_value=DataFrame({
        'time': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'tavg': [5, 10],
        'tmin': [0, 5],
        'tmax': [10, 15],
        'prcp': [0, 0.1],
        'snow': [0, 0],
        'wdir': [180, 270],
        'wspd': [10, 15],
        'wpgt': [20, 25],
        'pres': [1015, 1018],
        'tsun': [100, 200]
    })) as m:
        yield m


def test_get_coordinates(weather_data_class):
    # Test retrieval of coordinates
    assert weather_data_class.get_coordinates('JFK') == (40.6413, -73.7781)
    assert weather_data_class.get_coordinates('LHR') == (51.47, -0.4543)
    assert weather_data_class.get_coordinates('XYZ') == (None, None)


def test_assign_coordinates(weather_data_class, sample_flight_data):
    result_df = weather_data_class.assign_coordinates()
    # Check that coordinates are properly assigned
    assert 'dep_lat' in result_df
    assert 'dep_lon' in result_df
    assert 'arr_lat' in result_df
    assert 'arr_lon' in result_df
    assert result_df.loc[0, 'dep_lat'] == 40.6413
    assert result_df.loc[0, 'arr_lat'] == 37.6188056


def test_fetch_weather_data(mock_fetch, weather_data_class):
    # Assuming assign_coordinates should create the dep_lat and dep_lon columns
    weather_data_class.assign_coordinates()
    weather_data_class.df = DataFrame({
        'dep_lat': [50.8503, 50.8503],  # Example latitudes
        'dep_lon': [4.3517, 4.3517],  # Example longitudes
        'dep_iata_code': ['BRU', 'BRU'],
        'adt': [datetime(2023, 1, 1), datetime(2023, 1, 2)],  # Example arrival dates
        'aat': [datetime(2023, 1, 1), datetime(2023, 1, 2)],  # Example departure dates
        'arr_lat': [51.5074, 51.5074],  # Example arrival latitudes
        'arr_lon': [-0.1278, -0.1278],  # Example arrival longitudes
        'arr_iata_code': ['LON', 'LON']  # Example arrival IATA codes
    })
    weather_data_class.fetch_weather_data()


def test_merge_weather_with_flights(mock_fetch, weather_data_class):
    weather_data_class.assign_coordinates()
    weather_data_class.fetch_weather_data()

    # Merge and check output
    merged_df = weather_data_class.merge_weather_with_flights()
    print("Merged DataFrame:", merged_df.head())

    assert 'tavg_dep' in merged_df
    assert 'tavg_arr' in merged_df
    assert 'prcp_dep' in merged_df
    assert 'prcp_arr' in merged_df

    # Debug print to check NaN values in `tavg_dep`
    print("tavg_dep column null values:", merged_df['tavg_dep'].isnull().sum())
    print("Sample of merged tavg_dep values:", merged_df['tavg_dep'].head())

    assert merged_df['tavg_dep'].notnull().all(), "Some values in 'tavg_dep' are NaN"
    assert merged_df['tavg_arr'].notnull().all(), "Some values in 'tavg_arr' are NaN"