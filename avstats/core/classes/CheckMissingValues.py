import pandas as pd
from typing import Tuple

class DataCleaning:
    def __init__(self, unique_column: str) -> None:
        """
        Initialize the DataCleaning class with a unique column.

        Parameters:
        unique_column (str): The name of the column to check for duplicates.
        """
        self.unique_column = unique_column

    def check_missing_and_duplicates(self, df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
        """
        Check for missing values and duplicated rows in a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to check.

        Returns:
        tuple: A tuple containing:
            - missing_values (int): The total number of missing values.
            - duplicate_rows (pd.DataFrame): The DataFrame of duplicated rows.
        """
        if self.unique_column not in df.columns:
            raise ValueError(f"Column '{self.unique_column}' does not exist in DataFrame")

        missing_values = df.isna().sum().sum()
        print(f"Number of total missing values: {missing_values}")

        duplicate_rows = df[df.duplicated(subset=self.unique_column, keep=False)]
        print(f"Number of total duplicated rows: {len(duplicate_rows)}")

        missing_by_column = df.isnull().sum()
        print("Missing values by column:")
        print(missing_by_column)

        return missing_values, duplicate_rows
