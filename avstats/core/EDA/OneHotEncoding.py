# core/ML/OneHotEncoding.py
import pandas as pd
from typing import Any, List
from avstats.core.EDA.validators.validator_OneHotEncoding import OneHotEncodingInput


class OneHotEncoding:
    def __init__(self, df: pd.DataFrame, selected_columns: List[str]):
        """
        Initializes the OneHotEncoding class with the provided dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing data to encode.
        """
        # Validate input using Pydantic
        validated_input = OneHotEncodingInput(df=df)

        self.df = validated_input.df
        self.selected_columns = selected_columns

    def encode_selected_columns(self) -> tuple[Any, Any, Any] | Any:
        """
        Encodes all categorical columns dynamically using one-hot encoding,
        while excluding specified columns.

        Returns:
        pd.DataFrame: The dataframe with selected categorical columns one-hot encoded.
        """
        try:
            # Ensure only existing columns are used for encoding
            available_columns = [col for col in self.selected_columns if col in self.df.columns]

            if not available_columns:
                print("No valid categorical columns found for encoding.")

            # One-hot encode selected categorical columns
            df_encoded = pd.get_dummies(self.df, columns=available_columns, drop_first=False, prefix_sep="_",
                                        prefix='')

            # Remove leading underscores from column names
            df_encoded.columns = df_encoded.columns.str.lstrip('_')

            # Convert boolean columns (One-Hot Encoded) to numeric
            for col in df_encoded.select_dtypes(include=['bool']).columns:
                df_encoded[col] = df_encoded[col].astype(int)

            # Remove columns where all values are zero
            df_cleaned = df_encoded.loc[:, (df_encoded != 0).any(axis=0)].copy()

            # Ensure 'total_passengers' is retained and converted to numeric
            if 'total_passengers' in df_encoded.columns:
                df_cleaned['total_passengers'] = pd.to_numeric(df_cleaned['total_passengers'], errors='coerce')

            # Select only numeric columns
            df_numeric = df_cleaned.select_dtypes(include=['number'])

            return df_encoded, df_numeric, df_cleaned

        except KeyError as e:
            print(f"Encoding error: {e}")
            return self.df
