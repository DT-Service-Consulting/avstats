import pytest
import pandas as pd
from avstats.core.classes.CheckMissingValues import DataCleaning


@pytest.fixture
def sample_data():
    data = {
        'id': [1, 2, 2, 4, 5],
        'value': [10, 20, 20, None, 50]
    }
    df = pd.DataFrame(data)
    cleaner = DataCleaning(unique_column='id')
    return df, cleaner


@pytest.fixture
def clean_data(sample_data):
    df, cleaner = sample_data
    missing_values, duplicate_rows, missing_by_column = cleaner.check_missing_and_duplicates(df)
    return missing_values, duplicate_rows, missing_by_column


def test_check_missing_values(clean_data):
    missing_values, duplicate_rows, missing_by_column = clean_data
    assert missing_values == 1, "Should be 1 missing value"


def test_check_duplicate_rows(clean_data):
    missing_values, duplicate_rows, missing_by_column = clean_data
    assert len(duplicate_rows) == 2, "Should be 2 duplicated rows"


def test_return_types(clean_data):
    missing_values, duplicate_rows, missing_by_column = clean_data
    assert isinstance(missing_values, int), "Missing values should be an integer"
    assert isinstance(duplicate_rows, pd.DataFrame), "Duplicate rows should be a DataFrame"
