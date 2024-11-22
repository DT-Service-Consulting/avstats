import pytest
import pandas as pd
from avstats.core.EDA_workflow.NewFeatures import NewFeatures


@pytest.fixture
def sample_flight():
    return NewFeatures(
        uuid="12345",
        dep_delay=10,
        sdt=pd.Timestamp("2023-10-01 08:00:00"),
        sat=pd.Timestamp("2023-10-01 10:00:00"),
        cargo=False,
        private=False
    )


def test_categorize_flight(sample_flight):
    result = sample_flight.categorize_flight()
    expected = "Commercial"
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_time_window_departure(sample_flight):
    result = sample_flight.get_time_window(time_type='departure')
    expected = "Morning"
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_time_window_arrival(sample_flight):
    result = sample_flight.get_time_window(time_type='arrival')
    expected = "Morning"
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_time_window_invalid(sample_flight):
    with pytest.raises(ValueError, match="time_type must be either 'departure' or 'arrival'."):
        sample_flight.get_time_window(time_type='invalid')
