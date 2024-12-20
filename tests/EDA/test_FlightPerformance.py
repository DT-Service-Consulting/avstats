import pytest
import pandas as pd
from avstats.core.EDA_workflow.FlightPerformance import FlightPerformance


@pytest.fixture
def sample_data():
    """
    Provide a sample DataFrame for testing purposes.
    """
    data = {
        'dep_delay': [0, 15, 30, 45, None],
        'dep_delay_15': [0, 1, 1, 1, 0],
        'on_time_15': [1, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def flight_performance(sample_data):
    """
    Provide a FlightPerformance instance initialized with sample data.
    """
    return FlightPerformance(sample_data)

def test_overall_performance(flight_performance):
    """
    Test the overall_performance method.
    """
    result = flight_performance.overall_performance()

    assert 'Delayed Flights (%)' in result
    assert 'On-Time Flights (%)' in result
    assert 'Missing Status (%)' in result

    assert result['Delayed Flights (%)'] == pytest.approx(60.0, 0.1)
    assert result['On-Time Flights (%)'] == pytest.approx(20.0, 0.1)
    assert result['Missing Status (%)'] == pytest.approx(20.0, 0.1)

def test_delay_ranges_summary(flight_performance):
    """
    Test the delay_ranges_summary method with various ranges.
    """
    delay_ranges = [(0, 15), (16, 30), (31, 45)]
    result = flight_performance.delay_ranges_summary(delay_ranges)

    assert f"0-15 minutes" in result
    assert f"16-30 minutes" in result
    assert f"31-45 minutes" in result

    assert result["0-15 minutes"] == pytest.approx(20.0, 0.1)
    assert result["16-30 minutes"] == pytest.approx(20.0, 0.1)
    assert result["31-45 minutes"] == pytest.approx(20.0, 0.1)

def test_empty_dataframe():
    """
    Test methods with an empty DataFrame.
    """
    empty_df = pd.DataFrame({'dep_delay': [], 'dep_delay_15': [], 'on_time_15': []})
    performance = FlightPerformance(empty_df)

    overall_result = performance.overall_performance()
    assert overall_result['Delayed Flights (%)'] == 0
    assert overall_result['On-Time Flights (%)'] == 0
    assert overall_result['Missing Status (%)'] == 0

    delay_ranges = [(0, 15), (16, 30)]
    range_result = performance.delay_ranges_summary(delay_ranges)
    assert all(value == 0 for value in range_result.values())
