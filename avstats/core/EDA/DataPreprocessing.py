# DataPreprocessing.py
import pandas as pd
import numpy as np
from avstats.core.EDA.validators.validator_DataProcessing import DataProcessingInput


class DataPreprocessing:
    def __init__(self, df, unique_column: str) -> None:
        """
        Initialize the DataCleaning class with a unique column.

        Parameters:
        unique_column (str): The name of the column to check for duplicates.
        """
        # Validate inputs using Pydantic
        validated_input = DataProcessingInput(df=df, unique_column=unique_column)

        self.df = validated_input.df
        self.unique_column = validated_input.unique_column

    def check_missing_and_duplicates(self) -> dict:
        """
        Analyze the DataFrame for missing values and duplicates.

        Returns:
        dict: A dictionary containing:
            - 'missing_values': Total number of missing values.
            - 'duplicate_rows': DataFrame of duplicate rows (empty if none).
            - 'missing_by_column': Missing value counts for columns with missing values only.
        """
        if self.unique_column not in self.df.columns:
            raise ValueError(f"Column '{self.unique_column}' does not exist in the DataFrame")

        # Calculate missing values
        total_missing = self.df.isna().sum().sum()
        missing_percentage = (self.df.isna().mean() * 100).to_dict()
        missing_by_column = self.df.isnull().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]  # Filter only columns with missing values

        # Identify duplicate rows
        duplicate_rows = self.df[self.df.duplicated(subset=self.unique_column, keep=False)]
        duplicate_info = ("None" if duplicate_rows.empty else f"\n{duplicate_rows.head()}")

        return {
            "missing_values": total_missing,
            "missing_percentage": missing_percentage,
            "duplicate_rows": duplicate_info,
            "missing_by_column": missing_by_column,
        }

    def get_summary_statistics(self) -> dict:
        """
        Generate summary statistics for the DataFrame.

        Returns:
            dict: A dictionary containing:
                - 'numerical_summary': Summary statistics for numerical columns.
                - 'categorical_summary': Value counts for categorical columns.
                - 'data_types': Data types of all columns.
        """
        numerical_summary = self.df.describe(include=[np.number]).transpose()
        categorical_summary = {
            col: self.df[col].value_counts().to_dict()
            for col in self.df.select_dtypes(include=['object', 'category']).columns
        }
        data_types = self.df.dtypes.to_dict()

        return {
            "numerical_summary": numerical_summary,
            "categorical_summary": categorical_summary,
            "data_types": data_types,
        }

    def calculate_correlation(self) -> pd.DataFrame:
        """
        Calculate the correlation matrix for numerical columns.

        Returns:
            pd.DataFrame: Correlation matrix of numerical columns.
        """
        return self.df.corr()

    def check_data_balance(self) -> dict:
        """
        Analyze the balance of categorical variables.

        Returns:
            dict: A dictionary containing value counts for each categorical column.
        """
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        return {
            col: self.df[col].value_counts(normalize=True).to_dict()
            for col in categorical_columns
        }

    def detect_outliers(self, method: str = "IQR") -> pd.DataFrame:
        """
        Detect outliers in numerical columns.

        Parameters:
            method (str): Method to detect outliers ('IQR' or 'z-score').

        Returns:
            pd.DataFrame: A DataFrame highlighting outliers for each numerical column.
        """
        numerical_cols = self.df.select_dtypes(include=[np.number])
        outliers = pd.DataFrame(index=self.df.index)

        if method == "IQR":
            for col in numerical_cols:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers[col] = ~self.df[col].between(lower_bound, upper_bound)
        elif method == "z-score":
            for col in numerical_cols:
                mean = self.df[col].mean()
                std = self.df[col].std()
                outliers[col] = ((self.df[col] - mean) / std).abs() > 3
        else:
            raise ValueError("Invalid method. Choose 'IQR' or 'z-score'.")

        return outliers

    def exploratory_analysis(self) -> dict:
        """
        Perform exploratory data analysis (EDA) on the DataFrame.

        Returns:
            dict: Comprehensive report containing:
                - Summary statistics
                - Missing values analysis
                - Correlation analysis
                - Data balance check
        """
        summary = self.get_summary_statistics()
        missing_info = self.check_missing_and_duplicates()
        correlations = self.calculate_correlation()
        data_balance = self.check_data_balance()

        return {
            "summary_statistics": summary,
            "missing_values_analysis": missing_info,
            "correlation_matrix": correlations,
            "data_balance": data_balance,
        }

    def preprocess_data(self) -> pd.DataFrame:
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
        return self.df