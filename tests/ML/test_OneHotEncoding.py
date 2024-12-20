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

    # Initialize OneHotEncoding
    encoder = OneHotEncoding(df)

    # Call the encode_routes method
    encoded_df, corr_df, route_columns = encoder.encode_routes()

    # Check that the encoded dataframe is not None
    assert encoded_df is not None, "Encoded dataframe should not be None."

    # Check that route columns are correctly one-hot encoded
    expected_columns = ['total_dep_delay_15', 'A-B', 'B-C', 'C-D']
    assert all(col in encoded_df.columns for col in expected_columns), "Missing expected columns in encoded dataframe."

    # Check the scaling of dummy variables
    assert all(encoded_df['A-B'] == [2, 0, 2, 0]), "Scaling of dummy variables is incorrect."

    # Check that correlation dataframe includes the target and route columns
    assert all(col in corr_df.columns for col in ['total_dep_delay_15', 'A-B', 'B-C', 'C-D']), "Correlation dataframe is missing expected columns."

def test_clean_data():
    # Mock data
    data = {
        'total_passengers': ['100', '200', 'NaN', '400'],
        'A-B': [0, 0, 0, 0],
        'B-C': [2, 0, 2, 0]
    }
    df = pd.DataFrame(data)

    # Initialize OneHotEncoding
    encoder = OneHotEncoding(df)

    # Assign encoded dataframe directly for testing
    encoder.df_encoded = df

    # Call the clean_data method
    cleaned_df = encoder.clean_data()

    # Check that columns with all zeros are removed
    assert 'A-B' not in cleaned_df.columns, "Column with all zeros should be removed."

    # Check that 'total_passengers' is converted to numeric
    assert cleaned_df['total_passengers'].dtype in ['float64', 'int64'], "Column 'total_passengers' should be numeric."

    # Check the resulting dataframe contains numeric columns only
    assert all(cleaned_df.dtypes[col] in ['float64', 'int64'] for col in cleaned_df.columns), "Cleaned dataframe should contain numeric columns only."

if __name__ == "__main__":
    pytest.main()
