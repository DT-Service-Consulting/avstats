import pytest
import pandas as pd
from avstats.core.EDA.PassengerData import PassengerData


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame."""
    data = {
        'TIME': ['Airport1 - Airport2', 'Airport3 - Airport4', 'InvalidEntry'],
        'Passengers': [100, 200, None]
    }
    return pd.DataFrame(data)


@pytest.fixture
def airport_mapping():
    """Fixture providing a sample airport mapping."""
    return {
        'Airport1': 'A1',
        'Airport2': 'A2',
        'Airport3': 'A3',
        'Airport4': 'A4'
    }


def test_convert_to_route_code(airport_mapping):
    passenger_data = PassengerData(pd.DataFrame(), airport_mapping)

    # Valid airport entry
    result = passenger_data.convert_to_route_code('Airport1 - Airport2')
    assert result == 'A1-A2', "Route code conversion failed for valid input."

    # Invalid airport entry
    result = passenger_data.convert_to_route_code('Airport5 - Airport6')
    assert result == '-', "Route code conversion failed for unknown airports."

    # Incorrectly formatted input
    result = passenger_data.convert_to_route_code('Airport1')
    assert result is None, "Route code conversion failed for incorrectly formatted input."


def test_process_passenger_data(sample_dataframe, airport_mapping):
    passenger_data = PassengerData(sample_dataframe, airport_mapping)
    processed_df = passenger_data.process_passenger_data()

    # Assert 'route_code' column exists
    assert 'route_code' in processed_df.columns, "'route_code' column is missing after processing."

    # Assert rows with NaN are dropped
    assert processed_df.shape[0] == 2, "Rows with NaN values were not dropped."

    # Assert route codes are correctly assigned
    expected_route_codes = ['A1-A2', 'A3-A4']
    assert processed_df['route_code'].equals(pd.Series(expected_route_codes)), "Route codes are incorrect."

    # Assert 'AIRP_PR' column is removed
    assert 'AIRP_PR' not in processed_df.columns, "'AIRP_PR' column should be removed after processing."

    # Assert no rows with ":" exist
    assert not processed_df.apply(lambda row: row.astype(str).str.contains(":").any(), axis=1).any(), \
        "Rows containing ':' were not removed."


def test_process_passenger_data_invalid_data(airport_mapping):
    invalid_data = {
        'TIME': ['Invalid - Entry', 'Airport1 - Airport3', 'Airport1:Invalid', None],
        'Passengers': [50, 150, 30, 40]
    }
    df = pd.DataFrame(invalid_data)
    passenger_data = PassengerData(df, airport_mapping)
    processed_df = passenger_data.process_passenger_data()

    # Debugging: Print the resulting DataFrame
    print("\nProcessed DataFrame:")
    print(processed_df)

    # Assert valid rows remain
    assert processed_df.shape[0] == 1, "Invalid rows were not removed correctly."

    # Assert correct route code for valid entry
    assert processed_df['route_code'].iloc[0] == 'A1-A3', "Route code is incorrect for valid entry."

