from avstats.core.EDA_workflow.DataProcessing import DataProcessing
import pytest
import pandas as pd
from unittest.mock import patch


@pytest.fixture
def sample_dataframe():
    data = {
        "adt": ["2023-01-01 12:00:00", "2023-01-01 14:00:00"],
        "aat": ["2023-01-01 12:30:00", "2023-01-01 12:45:00"],
        "dep_delay": [15, 30],
        "sdt": ["2023-01-01 11:45:00", "2023-01-01 13:30:00"],
        "sat": ["2023-01-01 12:15:00", "2023-01-01 14:30:00"],
        "calc_sft": [100, 30],
        "calc_aft": [25, 70],
        "airline_iata_code": ["AA", "UA"],
        "flight_iata_number": ["123", "456"],
        "cargo": [1, 0],
        "private": [0, 1],
    }
    df = pd.DataFrame(data)
    return df

@patch("avstats.core.EDA_workflow.DataProcessing.NewFeatures")
def test_preprocess_avstats(mock_new_features, sample_dataframe):
    mock_new_features.categorize_flight.side_effect = lambda cargo, private: "Cargo" if cargo else "Private"
    mock_new_features.get_time_window.side_effect = lambda hour: "Morning" if 6 <= hour < 12 else "Afternoon"

    dp = DataProcessing(sample_dataframe, unique_column="flight_iata_number")
    processed_df = dp.preprocess_avstats()

    # Assert data types
    assert pd.api.types.is_datetime64_any_dtype(processed_df["adt"])
    assert pd.api.types.is_datetime64_any_dtype(processed_df["aat"])

    # Assert no missing values
    assert processed_df["dep_time_window"].isnull().sum() == 0
    assert processed_df["arr_time_window"].isnull().sum() == 0

    # Assert feature values
    assert processed_df["dep_time_window"].iloc[0] == "Afternoon"
    assert processed_df["arr_time_window"].iloc[0] == "Afternoon"
    assert processed_df["dep_time_window"].iloc[1] == "Afternoon"
    assert processed_df["arr_time_window"].iloc[1] == "Afternoon"



def test_check_missing_and_duplicates(sample_dataframe):
    dp = DataProcessing(sample_dataframe, unique_column="flight_iata_number")

    # Add duplicate row for testing
    duplicate_df = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]], ignore_index=True)

    result = dp.check_missing_and_duplicates(duplicate_df)

    # Assert missing values
    assert result["missing_values"] == 0

    # Assert duplicates
    assert result["duplicate_rows"] != "None"
    assert duplicate_df[duplicate_df.duplicated(subset="flight_iata_number", keep=False)].shape[0] > 0

