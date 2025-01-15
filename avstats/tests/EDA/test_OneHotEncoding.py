import pytest
import pandas as pd
from avstats.core.EDA.OneHotEncoding import OneHotEncoding


def test_encode_routes():
    # Mock data
    data = {
        'route_iata_code': ['A-B', 'B-C', 'A-B', 'C-D'],
        'total_dep_delay_15': [10, 15, 5, 0],
        'total_on_time_15': [1, 1, 0, 1],  # Add this column
    }
    df = pd.DataFrame(data)
    encoder = OneHotEncoding(df)
    result = encoder.encode_routes()
    assert result is not None, "encode_routes() should not return None."

    encoded_df = result
    assert encoded_df is not None, "Encoded dataframe should not be None."

    expected_columns = ['total_dep_delay_15', 'total_on_time_15', 'A-B', 'B-C', 'C-D']
    assert all(col in encoded_df.columns for col in expected_columns), "Missing expected columns in encoded dataframe."
    assert encoded_df['A-B'].tolist() == [2, 0, 2, 0], "Scaling of dummy variables is incorrect."


def test_clean_data():
    # Mock data with required columns
    data = {
        'total_passengers': ['100', '200', 'NaN', '400'],
        'A-B': [0, 0, 0, 0],
        'B-C': [2, 0, 2, 0],
        'route_iata_code': ['A-B', 'B-C', 'A-B', 'C-D'],  # Required column
        'total_dep_delay_15': [10, 15, 5, 0],  # Required column
        'total_on_time_15': [1, 0, 1, 1],  # Mock the missing column
    }
    df = pd.DataFrame(data)
    encoder = OneHotEncoding(df)
    encoder.encode_routes()  # Initialize `df_encoded` by running encode_routes()

    df_numeric, df_cleaned = encoder.clean_data()

    # Debugging: Print column types in df_numeric
    print("\nDebugging: df_numeric column types:")
    print(df_numeric.dtypes)

    # Assert columns with all zeros are removed
    assert 'A-B' in df_cleaned.columns, "Column 'A-B' should not be removed as it contains non-zero values."
    assert 'total_passengers' in df_cleaned.columns, "'total_passengers' column should be retained."

    # Assert `total_passengers` is converted to numeric
    assert df_cleaned['total_passengers'].dtype in ['float64', 'int64'], \
        "Column 'total_passengers' should be numeric."

    # Assert `df_numeric` contains only numeric columns
    assert all(dtype in ['float64', 'int64', 'int32'] for dtype in df_numeric.dtypes), \
        "Cleaned numeric dataframe should contain numeric columns only."


def test_missing_required_column():
    # Mock data without the required column
    data = {
        'total_dep_delay_15': [10, 15, 5, 0]
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match=r"Missing required columns: \{'route_iata_code'\}"):
        OneHotEncoding(df)


def test_uninitialized_encoded_dataframe():
    # Mock data with all required columns
    data = {
        'route_iata_code': ['A-B', 'B-C', 'A-B', 'C-D'],
        'total_dep_delay_15': [10, 15, 5, 0],
        'total_on_time_15': [1, 1, 1, 1],  # Add this column
    }
    df = pd.DataFrame(data)
    encoder = OneHotEncoding(df)
    with pytest.raises(ValueError, match="Encoded dataframe is not initialized. Run encode_routes\(\) first."):
        encoder.clean_data()
