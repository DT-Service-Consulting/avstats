import pandas as pd
from typing import Any


class DataCleaning:
    def __init__(self, unique_column: str) -> None:
        """
        Initialize the DataCleaning class with a unique column.

        Parameters:
        unique_column (str): The name of the column to check for duplicates.
        """
        self.unique_column = unique_column

    def check_missing_and_duplicates(self, df: pd.DataFrame) -> tuple[Any, Any, Any]:
        """
        Check for missing values and duplicated rows in a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to check.

        Returns:
        tuple: A tuple containing:
            - missing_values (int): The total number of missing values.
            - duplicate_rows (pd.DataFrame): The DataFrame of duplicated rows.
            - missing_by_column (pd.Series): The count of missing values by column.

        Example:
        >>> data = {
        ...     'id': [1, 2, 2, 4, 5],
        ...     'value': [10, 20, 20, None, 50]
        ... }
        >>> df = pd.DataFrame(data)
        >>> cleaner = DataCleaning(unique_column='id')
        >>> missing_values, duplicate_rows, missing_by_column = cleaner.check_missing_and_duplicates(df)
        >>> print(missing_values)
        1
        >>> print(duplicate_rows)
           id  value
        1   2   20.0
        2   2   20.0
        >>> print(missing_by_column)
        id       0
        value    1
        dtype: int64
        """

        if self.unique_column not in df.columns:
            raise ValueError(f"Column '{self.unique_column}' does not exist in DataFrame")

        missing_values = df.isna().sum().sum()
        duplicate_rows = df[df.duplicated(subset=self.unique_column, keep=False)]
        missing_by_column = df.isnull().sum()

        print(missing_values, missing_values.dtype)
        return missing_values, duplicate_rows, missing_by_column
