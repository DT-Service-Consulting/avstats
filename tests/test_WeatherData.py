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
    airports = {
        'JFK': {'lat': 40.6413, 'lon': -73.7781},
        'LHR': {'lat': 51.4700, 'lon': -0.4543},
        'SFO': {'lat': 37.6188056, 'lon': -122.3754167},
        'DXB': {'lat': 25.2532, 'lon': 55.3657}
    }
    weather_data = WeatherData(sample_flight_data)
    weather_data.airports = airports
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
        'tsun': [100, 200],
        'lat': [40.6413, 51.4700],
        'lon': [-73.7781, -0.4543],
        'iata_code': ['JFK', 'LHR']
    })) as m:
        yield m

def test_get_coordinates(weather_data_class):
    assert weather_data_class.get_coordinates('JFK') == (40.6413, -73.7781)
    assert weather_data_class.get_coordinates('XYZ') == (None, None)

def test_assign_coordinates(weather_data_class):
    result_df = weather_data_class.assign_coordinates()
    assert 'dep_lat' in result_df
    assert result_df.loc[0, 'dep_lat'] == 40.6413

def test_fetch_weather_data(mock_fetch, weather_data_class):
    weather_data_class.assign_coordinates()
    weather_data_class.fetch_weather_data()
    assert not weather_data_class.weather_df.empty

def test_merge_weather_with_flights(mock_fetch, weather_data_class):
    weather_data_class.assign_coordinates()
    weather_data_class.fetch_weather_data()
    merged_df = weather_data_class.merge_weather_with_flights()
    assert 'tavg_dep' in merged_df
