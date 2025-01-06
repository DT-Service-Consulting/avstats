import pytest
import pandas as pd
import numpy as np
from avstats.core.ML_workflow.DataPreparation import DataPreparation
from pydantic import ValidationError


@pytest.fixture
def sample_dataframe():
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "feature3": [5, 4, 3, 2, 1],
        "target": [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


def test_standardize_data(sample_dataframe):
    # Initialize DataPreparation instance
    dp = DataPreparation(sample_dataframe, target_variable="target")

    # Call the method
    x_scaled_df, y = dp.standardize_data()

    # Check that the scaled features have mean close to 0 and std close to 1
    assert np.allclose(x_scaled_df.mean(), 0, atol=1e-7), "Features are not standardized (mean != 0)"
    assert np.allclose(x_scaled_df.to_numpy().std(axis=0), 1, atol=1e-7), "Features are not standardized (std != 1)"

    # Check that the target variable is correctly returned
    assert y.equals(sample_dataframe["target"]), "Target variable mismatch"


def test_select_important_features(sample_dataframe):
    # Initialize DataPreparation instance
    dp = DataPreparation(sample_dataframe, target_variable="target")

    # Call standardize_data to set x and y
    dp.standardize_data()

    # Call the method
    x_important, important_features = dp.select_important_features(alpha=0.1, threshold_percentage=0.1)

    # Check that the output is a DataFrame
    assert isinstance(x_important, pd.DataFrame), "x_important is not a DataFrame"

    # Check that important_features is a Series
    assert isinstance(important_features, pd.Series), "important_features is not a Series"

    # Check that important features are selected correctly (non-zero coefficients)
    assert all(important_features != 0), "Important features contain zero coefficients"

    # Check that x_important only contains selected features
    assert list(x_important.columns) == list(important_features.index), "x_important columns do not match important_features"


def test_empty_dataframe():
    # Test with an empty dataframe
    empty_df = pd.DataFrame()
    with pytest.raises(ValidationError, match="DataFrame cannot be empty"):
        DataPreparation(empty_df, target_variable="target")


def test_no_target_column():
    # Test when target column is missing
    data = {"feature1": [1, 2, 3], "feature2": [4, 5, 6]}
    df = pd.DataFrame(data)
    with pytest.raises(ValidationError, match="Target variable 'target' is not present in the DataFrame columns"):
        DataPreparation(df, target_variable="target")


def test_invalid_dataframe_type():
    # Test when df is not a DataFrame
    invalid_df = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(ValidationError, match="Input must be a pandas DataFrame"):
        DataPreparation(invalid_df, target_variable="target")


def test_invalid_target_variable_type(sample_dataframe):
    # Test when target_variable is not a string
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        DataPreparation(sample_dataframe, target_variable=123)


def test_target_variable_not_in_df(sample_dataframe):
    # Test when target_variable is not in DataFrame columns
    with pytest.raises(ValidationError, match="Target variable 'nonexistent' is not present in the DataFrame columns"):
        DataPreparation(sample_dataframe, target_variable="nonexistent")
