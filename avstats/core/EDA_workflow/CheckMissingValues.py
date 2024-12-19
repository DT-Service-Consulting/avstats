# CheckMissingValues.py
import pandas as pd


class DataCleaning:
    def __init__(self, unique_column: str) -> None:
        """
        Initialize the DataCleaning class with a unique column.

        Parameters:
        unique_column (str): The name of the column to check for duplicates.
        """
        self.unique_column = unique_column

    def check_missing_and_duplicates(self, df: pd.DataFrame) -> dict:
        """
        Analyze the DataFrame for missing values and duplicates.

        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.

        Returns:
        dict: A dictionary containing:
            - 'missing_values': Total number of missing values.
            - 'duplicate_rows': DataFrame of duplicate rows (empty if none).
            - 'missing_by_column': Missing value counts for columns with missing values only.
        """
        if self.unique_column not in df.columns:
            raise ValueError(f"Column '{self.unique_column}' does not exist in the DataFrame")

        # Calculate missing values
        total_missing = df.isna().sum().sum()
        missing_by_column = df.isnull().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]  # Filter only columns with missing values

        # Identify duplicate rows
        duplicate_rows = df[df.duplicated(subset=self.unique_column, keep=False)]
        duplicate_info = ("None" if duplicate_rows.empty else f"\n{duplicate_rows.head()}")

        return {
            "missing_values": total_missing,
            "duplicate_rows": duplicate_info ,
            "missing_by_column": missing_by_column,
        }
