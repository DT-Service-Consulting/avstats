import pytest
import pandas as pd
from avstats.core.ML.Multicollinearity import Multicollinearity


@pytest.fixture
def sample_data():
    scaled_df = pd.DataFrame({
        "Feature1": [1, 2, 3, 4, 5],
        "Feature2": [2, 4, 6, 8, 10],
        "Feature3": [5, 6, 7, 8, 9],
    })
    y = pd.Series([1, 2, 3, 4, 5], name="Target")
    return scaled_df, y


def test_valid_multicollinearity_instance(sample_data):
    scaled_df, y = sample_data
    instance = Multicollinearity(scaled_df=scaled_df, y=y, verbose=True)
    assert isinstance(instance, Multicollinearity), "Instance should be of Multicollinearity class"


def test_empty_dataframe():
    with pytest.raises(ValueError, match="scaled_df cannot be empty"):
        Multicollinearity(scaled_df=pd.DataFrame(), y=pd.Series([1, 2, 3]), verbose=True)


def test_missing_values():
    scaled_df = pd.DataFrame({"Feature1": [1, None, 3]})
    with pytest.raises(ValueError, match="scaled_df contains missing values"):
        Multicollinearity(scaled_df=scaled_df, y=pd.Series([1, 2, 3]), verbose=True)


def test_invalid_target():
    scaled_df = pd.DataFrame({"Feature1": [1, 2, 3]})
    with pytest.raises(ValueError, match="Input should be an instance of Series"):
        Multicollinearity(scaled_df=scaled_df, y=[1, 2, 3], verbose=True)
