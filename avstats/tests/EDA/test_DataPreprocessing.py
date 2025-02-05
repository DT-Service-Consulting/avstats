import pytest
import math
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

    # **Reinitialize with duplicate dataset**
    dp = DataPreprocessing(duplicate_df, unique_column="flight_iata_number")

    result = dp.check_missing_and_duplicates()

    # Debugging: Print duplicates
    print("\nDetected Duplicate Rows:\n", result["duplicate_rows"])

    # Assert missing values
    expected_missing = duplicate_df.isna().sum().sum()  # Dynamically count missing values
    assert result["missing_values"] == expected_missing, (
        f"Expected {expected_missing} missing values, found {result['missing_values']}."
    )

    # **Fix Assertion for Duplicates**
    assert isinstance(result["duplicate_rows"], pd.DataFrame), "Duplicate rows should be a DataFrame."
    assert not result["duplicate_rows"].empty, "Duplicate rows were not detected."


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
    """Test outlier detection method."""
    df = sample_dataframe.copy()
    df["dep_delay"] = pd.to_numeric([15, 500], errors="coerce")  # Ensure numeric conversion
    print(f"Test DataFrame:\n{df}")

    processor = DataPreprocessing(df, unique_column="flight_iata_number")

    outliers = processor.detect_outliers(method="IQR", features=["dep_delay"], threshold=0.5)
    print(f"Outliers detected: {outliers}")

    assert "dep_delay" in outliers
    assert not outliers["dep_delay"].empty  # Ensure outliers are detected


def test_handle_outliers_remove(sample_dataframe):
    """Test removing detected outliers."""
    df = sample_dataframe.copy()
    df["dep_delay"] = [15, 500]  # Introduce an extreme outlier
    processor = DataPreprocessing(df, unique_column="flight_iata_number")

    df_cleaned, removed_percentage = processor.handle_outliers(
        method="remove", features=["dep_delay"], detection_method="IQR", threshold=0.5
    )

    # Debugging: Print results
    print(f"Original DF size: {len(df)}, Cleaned DF size: {len(df_cleaned)}, Removed %: {removed_percentage:.2f}%")
    print(f"Remaining dep_delay values: {df_cleaned['dep_delay'].values}")

    # Ensure outlier is removed
    assert 500 not in df_cleaned["dep_delay"].values, "Outlier 500 was not removed."
    assert len(df_cleaned) < len(df), "No rows were removed when they should have been."


def test_handle_outliers_cap(sample_dataframe):
    """Test capping detected outliers."""
    df = sample_dataframe.copy()
    df["dep_delay"] = [15, 500]  # Introduce an extreme outlier
    processor = DataPreprocessing(df, unique_column="flight_iata_number")

    df_capped, _ = processor.handle_outliers(method="cap", features=["dep_delay"], detection_method="IQR", threshold=0.5)

    q1 = df["dep_delay"].quantile(0.25)
    q3 = df["dep_delay"].quantile(0.75)
    iqr = q3 - q1
    raw_upper_bound = q3 + 0.5 * iqr
    expected_upper_bound = np.nextafter(raw_upper_bound, -np.inf)

    assert 500 not in df_capped["dep_delay"].values, "Outlier 500 was not capped."
    assert math.isclose(df_capped["dep_delay"].max(), expected_upper_bound, rel_tol=1e-12), (
        f"Max value {df_capped['dep_delay'].max()} does not match expected {expected_upper_bound} within tolerance.")


def test_check_time_consistency(sample_dataframe):
    """Test time consistency checks."""
    dp = DataPreprocessing(sample_dataframe, unique_column="flight_iata_number")
    updated_df, inconsistent_rows = dp.check_time_consistency()

    # Assert columns for time flags exist
    assert "early_departure_flag" in updated_df.columns, "'early_departure_flag' column is missing."
    assert "early_arrival_flag" in updated_df.columns, "'early_arrival_flag' column is missing."
    assert "inconsistency_flag" in updated_df.columns, "'inconsistency_flag' column is missing."

    # Assert no inconsistencies remain in the main DataFrame
    assert (updated_df["inconsistency_flag"] == 1).sum() == 0, "Main DataFrame still contains inconsistent rows."

    # Ensure inconsistent rows were actually removed
    assert len(inconsistent_rows) > 0, "Inconsistent rows were not properly identified before removal."
    assert inconsistent_rows["inconsistency_flag"].sum() == len(inconsistent_rows), "Not all flagged rows were removed."


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

    # Assert negative delays are set to zero
    assert (processed_df["dep_delay"] >= 0).all(), "Negative departure delays were not set to zero."

    # Assert negative `calc_sft` and `calc_aft` are converted to positive
    assert (processed_df["calc_sft"] >= 0).all(), "'calc_sft' contains negative values."
    assert (processed_df["calc_aft"] >= 0).all(), "'calc_aft' contains negative values."


def test_check_data_balance(sample_dataframe):
    """Test data balance checking."""
    dp = DataPreprocessing(sample_dataframe, unique_column="flight_iata_number")
    balance_df = dp.check_data_balance()

    # Assert balance DataFrame has required columns
    assert "column" in balance_df.columns, "'column' missing in balance DataFrame."
    assert "proportion" in balance_df.columns, "'proportion' missing in balance DataFrame."

    # Assert proportions sum to 1 for each categorical column
    for col in sample_dataframe.select_dtypes(include=['object', 'category']).columns:
        total_proportion = balance_df[balance_df["column"] == col]["proportion"].sum()
        assert np.isclose(total_proportion, 1), f"Proportions for {col} do not sum to 1."
