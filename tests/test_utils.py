import pytest
import pandas as pd
from avstats.core.flight_analysis_utils import calculate_time_window_percentages


@pytest.fixture
def sample_flight_data():
    data = {
        'dep_time_window': ['Morning', 'Morning', 'Afternoon', 'Evening', 'Evening', 'Evening'],
        'arr_time_window': ['Morning', 'Afternoon', 'Afternoon', 'Evening', 'Evening', 'Night']
    }
    return pd.DataFrame(data)


def test_calculate_time_window_percentages(sample_flight_data):
    result = calculate_time_window_percentages(sample_flight_data)

    expected_data = {
        'Time Window': ['Afternoon', 'Evening', 'Morning', 'Night'],
        'Departure Percentages (%)': [16.67, 50.0, 33.33, 0.0],
        'Arrival Percentages (%)': [33.33, 33.33, 16.67, 16.67]
    }
    expected_df = pd.DataFrame(expected_data).set_index('Time Window')

    pd.testing.assert_frame_equal(result, expected_df)