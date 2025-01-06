# DataProcessing.py
import pandas as pd
from avstats.core.EDA_workflow.NewFeatures import NewFeatures


class DataProcessing:
    def __init__(self, df, unique_column: str) -> None:
        """
        Initialize the DataCleaning class with a unique column.

        Parameters:
        unique_column (str): The name of the column to check for duplicates.
        """
        self.df = df
        self.unique_column = unique_column

    def preprocess_avstats(self) -> pd.DataFrame:
        """
        Preprocess the aviation statistics DataFrame by cleaning and engineering features.

        This method handles missing values, calculates additional features, and ensures proper datetime conversions.

        Returns:
            pd.DataFrame: The processed DataFrame with additional features and cleaned data.
        """
        self.df['adt'] = pd.to_datetime(self.df['adt'], errors='coerce')
        self.df['aat'] = pd.to_datetime(self.df['aat'], errors='coerce')

        # Discard records with null dep_delay unless both sdt and adt are available
        self.df = self.df[~(self.df['dep_delay'].isnull() & (self.df['sdt'].isnull() | self.df['adt'].isnull()))].copy()
        datetime_columns = ['sat', 'sdt', 'aat', 'adt']
        self.df[datetime_columns] = self.df[datetime_columns].apply(pd.to_datetime, errors='coerce')

        # Calculate dep_delay if missing
        mask = self.df['dep_delay'].isnull() & self.df['sdt'].notnull() & self.df['adt'].notnull()
        self.df.loc[mask, 'dep_delay'] = (self.df['adt'] - self.df['sdt']).dt.total_seconds() / 60
        self.df['dep_delay'] = self.df['dep_delay'].fillna(0)

        # Impute other missing values
        self.df['adt'] = self.df['adt'].fillna(self.df['sdt'] + pd.to_timedelta(self.df['dep_delay'], unit='m'))
        self.df['aat'] = self.df['aat'].fillna(self.df['sat'] + pd.to_timedelta(self.df['dep_delay'], unit='m'))
        self.df['calc_sft'] = self.df['calc_sft'].fillna((self.df['sat'] - self.df['sdt']) / pd.Timedelta(minutes=1))
        self.df['calc_aft'] = self.df['calc_aft'].fillna((self.df['aat'] - self.df['adt']) / pd.Timedelta(minutes=1))
        self.df.fillna({'airline_iata_code': 'NONE', 'flight_iata_number': 'NONE'}, inplace=True)

        # Feature engineering
        self.df['dep_delay_15'] = (self.df['dep_delay'] > 15).astype(int)
        self.df['dep_delay_cat'] = pd.cut(
            self.df['dep_delay'], bins=[-float('inf'), 15, 60, float('inf')], labels=['Short', 'Medium', 'Long'])

        self.df['flight_cat'] = self.df.apply(
            lambda row: NewFeatures.categorize_flight(row['cargo'], row['private']), axis=1)

        self.df['dep_time_window'] = self.df['adt'].apply(
            lambda x: NewFeatures.get_time_window(x.hour) if pd.notnull(x) else None)

        self.df['arr_time_window'] = self.df['aat'].apply(
            lambda x: NewFeatures.get_time_window(x.hour) if pd.notnull(x) else None)

        self.df['on_time_15'] = (self.df['dep_delay'] < 15).astype(int)

        return self.df

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
            "duplicate_rows": duplicate_info,
            "missing_by_column": missing_by_column,
        }
