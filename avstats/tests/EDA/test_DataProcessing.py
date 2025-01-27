from avstats.core.EDA.DataPreprocessing import DataProcessing
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
        "calc_sft": [100, 50],  # Ensure no missing values
        "calc_aft": [25, 70],   # Ensure no missing values
        "airline_iata_code": ["AA", "UA"],
        "flight_iata_number": ["123", "456"],
        "cargo": [1, 0],
        "private": [0, 1],
    }
    df = pd.DataFrame(data)
    return df


@patch("avstats.core.EDA.DataProcessing.NewFeatures")
def test_preprocess_avstats(mock_new_features, sample_dataframe):
    # Mock NewFeatures methods
    mock_new_features.categorize_flight.side_effect = lambda cargo, private: "Cargo" if cargo else "Private"
    mock_new_features.get_time_window.side_effect = lambda hour: "Morning" if 6 <= hour < 12 else "Afternoon"

    dp = DataProcessing(sample_dataframe, unique_column="flight_iata_number")
    processed_df = dp.preprocess_avstats()

    # Assert data types
    assert pd.api.types.is_datetime64_any_dtype(processed_df["adt"]), "'adt' should be datetime64 dtype"
    assert pd.api.types.is_datetime64_any_dtype(processed_df["aat"]), "'aat' should be datetime64 dtype"

    # Assert no missing values in critical datetime columns
    assert processed_df["adt"].isnull().sum() == 0, "Missing values in 'adt'"
    assert processed_df["aat"].isnull().sum() == 0, "Missing values in 'aat'"

    # Assert missing values were imputed
    assert processed_df["calc_sft"].isnull().sum() == 0, "Missing values in 'calc_sft' not filled"
    assert processed_df["calc_aft"].isnull().sum() == 0, "Missing values in 'calc_aft' not filled"


@patch("avstats.core.EDA.DataProcessing.NewFeatures")
def test_feature_engineering(mock_new_features, sample_dataframe):
    # Mock NewFeatures methods
    mock_new_features.categorize_flight.side_effect = lambda cargo, private: "Cargo" if cargo else "Private"
    mock_new_features.get_time_window.side_effect = lambda hour: "Morning" if 6 <= hour < 12 else "Afternoon"

    dp = DataProcessing(sample_dataframe, unique_column="flight_iata_number")
    dp.preprocess_avstats()  # Preprocess the data first
    engineered_df = dp.feature_engineering()

    # Assert new features exist
    assert "dep_delay_15" in engineered_df.columns, "'dep_delay_15' column is missing"
    assert "dep_delay_cat" in engineered_df.columns, "'dep_delay_cat' column is missing"
    assert "flight_cat" in engineered_df.columns, "'flight_cat' column is missing"
    assert "dep_time_window" in engineered_df.columns, "'dep_time_window' column is missing"
    assert "arr_time_window" in engineered_df.columns, "'arr_time_window' column is missing"

    # Assert feature values
    assert list(engineered_df["dep_delay_15"]) == [0, 1], "'dep_delay_15' is incorrect"
    assert list(engineered_df["dep_delay_cat"]) == ["Short", "Medium"], "'dep_delay_cat' is incorrect"
    assert list(engineered_df["flight_cat"]) == ["Cargo", "Private"], "'flight_cat' is incorrect"
    assert list(engineered_df["dep_time_window"]) == ["Afternoon", "Afternoon"], "'dep_time_window' is incorrect"
    assert list(engineered_df["arr_time_window"]) == ["Afternoon", "Afternoon"], "'arr_time_window' is incorrect"


def test_check_missing_and_duplicates(sample_dataframe):
    dp = DataProcessing(sample_dataframe, unique_column="flight_iata_number")

    # Add duplicate row for testing
    duplicate_df = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]], ignore_index=True)

    print("\nDuplicate DataFrame:")
    print(duplicate_df)
    print("Missing values by column:")
    print(duplicate_df.isna().sum())

    result = dp.check_missing_and_duplicates(duplicate_df)

    # Assert no unexpected missing values
    expected_missing = 0
    assert result["missing_values"] == expected_missing, f"There should be {expected_missing} missing values, found {result['missing_values']}."

    # Assert duplicates
    assert result["duplicate_rows"] != "None", "There should be duplicate rows detected."
    assert duplicate_df[duplicate_df.duplicated(subset="flight_iata_number", keep=False)].shape[0] > 0
