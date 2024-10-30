import pytest
import pandas as pd
from avstats.core.classes.Multicollinearity import Multicollinearity


@pytest.fixture
def sample_data():
    data = {
        'target': [1, 2, 3, 4, 5],
        'feature1': [4, 2, 1, 3, 5],
        'feature2': [4, 2, 1, 3, 5],
        'feature3': [5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def multicollinearity(sample_data):
    return Multicollinearity(sample_data)


def test_calculate_vif(multicollinearity):
    result = multicollinearity.calculate_vif()
    expected_features = ['target', 'feature1', 'feature2', 'feature3']
    assert list(result[
                    'feature']) == expected_features, f"Expected: {expected_features}, got {list(result['feature'])}"
    assert all(result['VIF'] >= 1), "VIF values should be greater than or equal to 1"


def test_remove_high_vif_features(multicollinearity):
    result = multicollinearity.remove_high_vif_features(target_variable='target', threshold=5)
    expected_columns = ['target', 'feature3']
    assert list(
        result.columns) == expected_columns, f"Expected columns {expected_columns}, but got {list(result.columns)}"
