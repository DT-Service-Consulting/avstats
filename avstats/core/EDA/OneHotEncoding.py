# core/ML/OneHotEncoding.py
import pandas as pd
from typing import Any, Union
from avstats.core.EDA.validators.validator_OneHotEncoding import OneHotEncodingInput


class OneHotEncoding:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the OneHotEncoding class with the provided dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing data to encode.
        """
        # Validate input using Pydantic
        validated_input = OneHotEncodingInput(df=df)

        self.df = validated_input.df
        self.df = df
        self.df_encoded = None

    def encode_routes(self) -> Union[pd.DataFrame, None]:
        """
        Encodes the specified route column, modifies the dummy variables,
        and creates a subset of the dataframe for correlation analysis.

        Returns:
            Union[pd.DataFrame, None]: The full dataframe with encoded route columns.

        Raises:
        KeyError: If required columns are not found in the dataframe.
        """
        try:
            # One-hot encode the 'route_iata_code' column
            df_encoded = pd.get_dummies(self.df, columns=['route_iata_code'], drop_first=False, prefix='')

            # Convert dummy variables to numerical format (0 to 0, 1 to 2)
            for col in df_encoded.columns:
                if '-' in col:
                    df_encoded[col] = df_encoded[col] * 2

            # Remove leading underscores from column names
            df_encoded.columns = df_encoded.columns.str.lstrip('_')

            # Store encoded dataframe for further processing
            self.df_encoded = df_encoded

            return self.df_encoded

        except KeyError as e:
            print(f"Encoding error: {e}")
            return None

    def clean_data(self) -> tuple[Any, Any]:
        """
        Removes columns in the encoded dataframe where all values are zero
        and converts 'total_passengers' to numeric format.

        Returns:
        pd.DataFrame: The dataframe with zero-only columns removed and numeric columns retained.

        Raises:
        KeyError: If the 'total_passengers' column is missing from the dataframe.
        """
        if self.df_encoded is None:
            raise ValueError("Encoded dataframe is not initialized. Run encode_routes() first.")

        # Remove columns where all values are zero
        df_cleaned = self.df_encoded.loc[:, (self.df_encoded != 0).any(axis=0)].copy()

        # Ensure 'total_passengers' is retained
        if 'total_passengers' in self.df_encoded.columns:
            # Convert 'total_passengers' to numeric, coercing errors to NaN
            df_cleaned['total_passengers'] = pd.to_numeric(df_cleaned['total_passengers'], errors='coerce')

        # Select only numeric columns
        df_numeric = df_cleaned.select_dtypes(include=['number'])

        return df_numeric, df_cleaned
