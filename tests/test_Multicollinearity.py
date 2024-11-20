import pytest
import pandas as pd
from avstats.core.ML_workflow.Multicollinearity import Multicollinearity


@pytest.fixture
def sample_data():
    # Create a simple DataFrame with correlated features and a target variable
    data = {
        'A': [1, 0, 1, 0, 1],
        'B': [1, 0, 1, 1, 1],  # Some correlation with A
        'C': [1, 1, 0, 0, 0],  # Another feature
        'D': [1, 0, 1, 0, 0],  # Some correlation with A
        'E': [1, 0, 1, 0, 1],  # Perfectly correlated with A - Removed
        'F': [1, 1, 1, 1, 1]   # Constant feature - Removed
    }
    df = pd.DataFrame(data)
    y = pd.Series([1, 0, 1, 0, 1], name='target')
    return df, y


def test_remove_high_vif_features(sample_data):
    df, y = sample_data
    multi = Multicollinearity(scaled_df=df, y=y, verbose=False)
    result_df, features = multi.remove_high_vif_features(threshold=5)

    assert 'A' not in features.columns, "Feature 'A' with high VIF removed"
    assert 'B' in features.columns,     "Feature 'B' with mid VIF not removed"
    assert 'C' in features.columns,     "Feature 'C' with low VIF not removed"
    assert 'D' in features.columns,     "Feature 'D' with mid VIF not removed"
    assert 'E' not in features.columns, "Feature 'E' with high VIF removed"
    assert 'F' not in features.columns, "Feature 'F' constant removed"

    print(result_df)
