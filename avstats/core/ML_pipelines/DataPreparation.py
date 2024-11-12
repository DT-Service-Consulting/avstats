# core/ML_pipelines/DataPreparation.py
import pandas as pd

class DataPreparation:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df_encoded = None
        self.df_reduced = None

    def encode_routes(self):
        """
        Encodes the specified route column, modifies the dummy variables,
        and creates a subset of the dataframe for correlation analysis.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to encode.
        route_column (str): The name of the column to one-hot encode.
        target_column (str): The target column for correlation analysis.

        Returns:
        pd.DataFrame: A subset of the dataframe with encoded route columns and the target column.
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

            # Select the route code columns and the target column
            route_columns = [col for col in df_encoded.columns if '-' in col]
            corr_columns = ['total_dep_delay_15'] + route_columns

            # Create a subset of the dataframe for correlation
            corr_df = df_encoded[corr_columns]

            # Store encoded dataframe for further processing
            self.df_encoded = df_encoded

            return df_encoded, corr_df, route_columns

        except KeyError as e:
            print(f"Encoding error: {e}")
            return None

    def clean_data(self):
        """
        Removes columns in the encoded dataframe where all values are zero.

        Returns:
        pd.DataFrame: The dataframe with zero-only columns removed.
        """
        df_reduced = self.df_encoded.loc[:, (self.df_encoded != 0).any(axis=0)]
        self.df_reduced = df_reduced

        return df_reduced

    def prepare_features(self):
        """
        Converts 'total_passengers' to numeric, drops specified features,
        and returns the reduced dataframe with relevant numeric features.

        Returns:
        pd.DataFrame: The reduced dataframe with selected features.
        """
        # Convert 'total_passengers' to numeric, coercing errors to NaN
        self.df_reduced.loc[:, 'total_passengers'] = pd.to_numeric(self.df_reduced['total_passengers'], errors='coerce')

        # Select only numeric columns
        df_numeric = self.df_reduced.select_dtypes(include=['number'])

        # List of features to drop
        features_to_drop = ['total_flights', 'total_dep_delay_15', 'departures', 'arrivals']

        # Drop the specified columns
        df_numeric = df_numeric.drop(columns=features_to_drop)

        return df_numeric