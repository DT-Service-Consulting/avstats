import pytest
import pandas as pd
import numpy as np
from avstats.core.EDA.DataPreprocessing import DataPreprocessing


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        "adt": ["2023-01-01 12:00:00", "2023-01-01 14:00:00"],
        "aat": ["2023-01-01 12:30:00", "2023-01-01 12:45:00"],
        "dep_delay": [15, None],
        "sdt": ["2023-01-01 11:45:00", None],
        "sat": ["2023-01-01 12:15:00", None],
        "calc_sft": [100, None],
        "calc_aft": [25, None],
        "airline_iata_code": ["AA", "UA"],
        "flight_iata_number": ["123", "456"],
        "cargo": [1, 0],
        "private": [0, 1],
    }
    return pd.DataFrame(data)


def test_check_missing_and_duplicates(sample_dataframe):
    """Test for missing values and duplicate detection."""
    dp = DataPreprocessing(sample_dataframe, unique_column="flight_iata_number")

    # Add duplicate row for testing
    duplicate_df = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]], ignore_index=True)

    result = dp.check_missing_and_duplicates()

    # Assert missing values
    expected_missing = 5  # Manually count missing in the fixture
    assert result["missing_values"] == expected_missing, (
        f"Expected {expected_missing} missing values, found {result['missing_values']}."
    )

    # Assert duplicate rows
    assert result["duplicate_rows"] != "None", "Duplicate rows were not detected."


def test_get_summary_statistics(sample_dataframe):
    """Test summary statistics generation."""
    dp = DataPreprocessing(sample_dataframe, unique_column="flight_iata_number")
    summary = dp.get_summary_statistics()

    # Assert numerical summary exists and contains expected columns
    assert isinstance(summary["numerical_summary"], pd.DataFrame), "Numerical summary is not a DataFrame."
    assert "dep_delay" in summary["numerical_summary"].index, "'dep_delay' not included in numerical summary."

    # Assert categorical summary
    assert "airline_iata_code" in summary["categorical_summary"], "'airline_iata_code' missing in categorical summary."

    # Assert data types
    assert summary["data_types"]["adt"] == np.dtype("O"), "'adt' data type mismatch."


def test_detect_outliers(sample_dataframe):
    """Test outlier detection."""
    dp = DataPreprocessing(sample_dataframe, unique_column="flight_iata_number")

    outliers = dp.detect_outliers(features=["dep_delay", "calc_sft"], method="IQR", threshold=1.5)

    # Check that outliers for "calc_sft" are empty (as there are no outliers in the sample data)
    assert outliers["calc_sft"].empty, "Unexpected outliers detected in 'calc_sft'."


def test_handle_outliers(sample_dataframe):
    """Test outlier handling."""
    dp = DataPreprocessing(sample_dataframe, unique_column="flight_iata_number")

    # Inject outliers
    dp.df.loc[0, "dep_delay"] = 1000  # Add an extreme value
    cleaned_df = dp.handle_outliers(method="remove", features=["dep_delay"], detection_method="IQR", threshold=1.5)

    # Assert extreme outliers are removed
    assert cleaned_df["dep_delay"].max() <= 100, "Outliers not removed correctly."


def test_check_time_consistency(sample_dataframe):
    """Test time consistency checks."""
    dp = DataPreprocessing(sample_dataframe, unique_column="flight_iata_number")
    updated_df, inconsistent_rows = dp.check_time_consistency()

    # Assert columns for time flags exist
    assert "early_departure_flag" in updated_df.columns, "'early_departure_flag' column is missing."
    assert "early_arrival_flag" in updated_df.columns, "'early_arrival_flag' column is missing."
    assert "inconsistency_flag" in updated_df.columns, "'inconsistency_flag' column is missing."

    # Assert no inconsistencies remain in the main DataFrame
    assert inconsistent_rows.empty, "Inconsistent rows were not removed."


def test_preprocess_data(sample_dataframe):
    """Test the main preprocessing method."""
    dp = DataPreprocessing(sample_dataframe, unique_column="flight_iata_number")
    processed_df = dp.preprocess_data()

    # Assert data types are correctly converted
    assert pd.api.types.is_datetime64_any_dtype(processed_df["adt"]), "'adt' should be datetime64 dtype."
    assert pd.api.types.is_datetime64_any_dtype(processed_df["aat"]), "'aat' should be datetime64 dtype."

    # Assert missing values are imputed
    assert processed_df["calc_sft"].isnull().sum() == 0, "'calc_sft' missing values were not filled."
    assert processed_df["calc_aft"].isnull().sum() == 0, "'calc_aft' missing values were not filled."
