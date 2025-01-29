# DataPreprocessing.py
import pandas as pd
import numpy as np
from pandas import DataFrame
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

    def check_data_balance(self) -> pd.DataFrame:
        """
        Analyze the balance of categorical variables.

        Returns:
            pd.DataFrame: A DataFrame containing value counts for each categorical column.
                          Each row represents a category value, and columns represent
                          the categorical variable, value, and proportion.
        """
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        balance_data = []

        for col in categorical_columns:
            value_counts = self.df[col].value_counts(normalize=True).reset_index()
            value_counts.columns = [col + "_value", "proportion"]
            value_counts["column"] = col
            balance_data.append(value_counts)

        balance_df = pd.concat(balance_data, ignore_index=True)

        return balance_df

    def detect_outliers(self, method="IQR", features=None, threshold=1.5):
        """
        Detect outliers in the dataset using IQR or z-score.

        Parameters:
        - method (str): The outlier detection method ("IQR" or "z-score").
        - features (list): List of numerical features to check for outliers.
        - threshold (float): Threshold for outlier detection (1.5 for IQR, 3 for z-score).

        Returns:
        - pd.DataFrame: DataFrame containing outliers for the specified features.
        """
        if features is None:
            features = self.df.select_dtypes(include=[np.number]).columns

        outliers = {}
        for feature in features:
            if method == "IQR":
                q1 = self.df[feature].quantile(0.25)
                q3 = self.df[feature].quantile(0.75)
                iqr = q3 - q1
                mask = (self.df[feature] < (q1 - threshold * iqr)) | \
                       (self.df[feature] > (q3 + threshold * iqr))
            elif method == "z-score":
                mean = self.df[feature].mean()
                std = self.df[feature].std()
                mask = np.abs((self.df[feature] - mean) / std) > threshold
            else:
                raise ValueError("Invalid method. Choose 'IQR' or 'z-score'.")

            outliers[feature] = self.df[mask]

        return outliers

    def handle_outliers(self, method="remove", features=None, detection_method="IQR", threshold=1.5):
        outliers = self.detect_outliers(method=detection_method, features=features, threshold=threshold)
        """
        Handle outliers in the dataset.

        Parameters:
        - method (str): How to handle outliers ("remove" or "cap").
        - features (list): List of numerical features to handle outliers.
        - detection_method (str): Outlier detection method ("IQR" or "z-score").
        - threshold (float): Threshold for outlier detection.

        Returns:
        - pd.DataFrame: Updated DataFrame after handling outliers.
        """
        for feature, outlier_df in outliers.items():
            if method == "remove":
                # Align indices before dropping
                common_indices = self.df.index.intersection(outlier_df.index)
                self.df = self.df.drop(common_indices)
            elif method == "cap":
                if detection_method == "IQR":
                    q1 = self.df[feature].quantile(0.25)
                    q3 = self.df[feature].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                elif detection_method == "z-score":
                    mean = self.df[feature].mean()
                    std = self.df[feature].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                else:
                    raise ValueError("Invalid detection method. Use 'IQR' or 'z-score'.")

                self.df[feature] = self.df[feature].clip(lower=lower_bound, upper=upper_bound)

        return self.df

    def check_time_consistency(self) -> tuple[DataFrame, DataFrame]:
        """
        Checks for inconsistencies in time columns (ADT, SDT, AAT, SAT) and flags early departures/arrivals.

        Returns:
            pd.DataFrame: Subset of the DataFrame with flagged inconsistencies and early departure/arrival indicators.
        """
        total_flights = len(self.df)

        # Flag ADT before SDT as "early departures"
        self.df['early_departure_flag'] = (self.df['adt'] < self.df['sdt']).astype(int)
        early_departures = self.df[self.df['early_departure_flag'] == 1]
        early_departures_count = len(early_departures)
        early_departures_percentage = (early_departures_count / total_flights) * 100

        # Flag AAT before SAT as "early arrivals"
        self.df['early_arrival_flag'] = (self.df['aat'] < self.df['sat']).astype(int)
        early_arrivals = self.df[self.df['early_arrival_flag'] == 1]
        early_arrivals_count = len(early_arrivals)
        early_arrivals_percentage = (early_arrivals_count / total_flights) * 100

        # Check for ADT after AAT
        self.df['inconsistency_flag'] = (self.df['adt'] > self.df['aat']).astype(int)
        inconsistent_flights = self.df[self.df['inconsistency_flag'] == 1]
        inconsistent_flights_count = len(inconsistent_flights)
        inconsistent_rows = pd.concat([inconsistent_flights])
        inconsistent_flights_percentage = (inconsistent_flights_count / total_flights) * 100

        # Remove flights with inconsistent times
        self.df = self.df[~(self.df['inconsistency_flag'] == 1)]

        # Print summary
        print(f"Early Departures: {early_departures_count} ({early_departures_percentage:.2f}%)")
        print(f"Early Arrivals: {early_arrivals_count} ({early_arrivals_percentage:.2f}%)")
        print(f"Inconsistent ADT > AAT: {inconsistent_flights_count} ({inconsistent_flights_percentage:.2f}%)")
        print(f"Dataset after removing inconsistent flights: {len(self.df)} rows")

        return self.df, inconsistent_rows

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
