import pytest
import pandas as pd
from avstats.core.classes.FlightPerformance import FlightPerformance


@pytest.fixture
def sample_flight_data():
    data = {
        'uuid': ['1', '2', '3', '4', '5'],
        'dep_delay_15': [0, 0, 1, 1, 0],
        'on_time_15': [1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def flight_performance(sample_flight_data):
    return FlightPerformance(sample_flight_data)


def test_overall_performance(flight_performance):
    result = flight_performance.overall_performance()
    expected = {
        "Delayed Flights": 40.0,
        "On-Time Flights": 60.0,
        "Flights with Missing Status": 0.0
    }
    assert result == expected, f"Expected {expected}, but got {result}"


def test_delayed_flight_percentages(flight_performance):
    delay_ranges = [(0, 1, "0-1 min"), (1, 2, "1-2 min")]
    result = flight_performance.delayed_flight_percentages(delay_ranges)
    expected = {
        "0-1 min": 40.0,
        "1-2 min": 0.0
    }
    assert result == expected, f"Expected {expected}, but got {result}"

