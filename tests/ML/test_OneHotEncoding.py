import pytest
import pandas as pd
from avstats.core.ML_workflow.OneHotEncoding import OneHotEncoding

def test_encode_routes():
    # Mock data
    data = {
        'route_iata_code': ['A-B', 'B-C', 'A-B', 'C-D'],
        'total_dep_delay_15': [10, 15, 5, 0]
    }
    df = pd.DataFrame(data)
    encoder = OneHotEncoding(df)
    result = encoder.encode_routes()
    assert result is not None, "encode_routes() should not return None."

    encoded_df, corr_df, route_columns = result
    assert encoded_df is not None, "Encoded dataframe should not be None."

    expected_columns = ['total_dep_delay_15', 'A-B', 'B-C', 'C-D']
    assert all(col in encoded_df.columns for col in expected_columns), "Missing expected columns in encoded dataframe."
    assert all(encoded_df['A-B'] == [2, 0, 2, 0]), "Scaling of dummy variables is incorrect."
    assert all(col in corr_df.columns for col in expected_columns), "Correlation dataframe is missing expected columns."
    assert route_columns == ['A-B', 'B-C', 'C-D'], "Route columns list is incorrect."


def test_clean_data():
    # Mock data with required columns
    data = {
        'total_passengers': ['100', '200', 'NaN', '400'],
        'A-B': [0, 0, 0, 0],
        'B-C': [2, 0, 2, 0],
        'route_iata_code': ['A-B', 'B-C', 'A-B', 'C-D'],  # Required column
        'total_dep_delay_15': [10, 15, 5, 0]  # Required column
    }
    df = pd.DataFrame(data)
    encoder = OneHotEncoding(df)
    encoder.df_encoded = df
    cleaned_df = encoder.clean_data()
    assert 'A-B' not in cleaned_df.columns, "Column with all zeros should be removed."
    assert cleaned_df['total_passengers'].dtype in ['float64', 'int64'], \
        "Column 'total_passengers' should be numeric."
    assert all(cleaned_df.dtypes[col] in ['float64', 'int64'] for col in cleaned_df.columns), \
        "Cleaned dataframe should contain numeric columns only."


def test_missing_required_column():
    # Mock data without the required column
    data = {
        'total_dep_delay_15': [10, 15, 5, 0]
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match=r"Missing required columns: \{'route_iata_code'\}"):
        OneHotEncoding(df)


def test_uninitialized_encoded_dataframe():
    # Mock data
    data = {
        'route_iata_code': ['A-B', 'B-C', 'A-B', 'C-D'],
        'total_dep_delay_15': [10, 15, 5, 0]
    }
    df = pd.DataFrame(data)
    encoder = OneHotEncoding(df)
    with pytest.raises(ValueError, match="Encoded dataframe is not initialized. Run encode_routes\(\) first."):
        encoder.clean_data()
