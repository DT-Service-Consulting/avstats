# core/ML_workflow/DataPreparation.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

class DataPreparation:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df_encoded = None
        self.df_numeric = None
        self.x = None
        self.y = None

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
        Removes columns in the encoded dataframe where all values are zero
        and converts 'total_passengers' to numeric

        Returns:
        pd.DataFrame: The dataframe with zero-only columns removed.
        """
        df_reduced = self.df_encoded.loc[:, (self.df_encoded != 0).any(axis=0)]

        # Convert 'total_passengers' to numeric, coercing errors to NaN
        df_reduced.loc[:, 'total_passengers'] = pd.to_numeric(df_reduced['total_passengers'], errors='coerce')

        # Select only numeric columns
        self.df_numeric = df_reduced.select_dtypes(include=['number'])

        return self.df_numeric

    def standardize_data(self):
        # Prepare data
        self.x = self.df_numeric.drop(columns=['total_dep_delay'])  # drop the target variable
        self.y = self.df_numeric['total_dep_delay']

        # Create the scaler instance
        scaler = StandardScaler()

        # Apply scaling to the features
        x_scaled = scaler.fit_transform(self.x)

        # Convert the scaled data back into a DataFrame for easier handling
        x_scaled_df = pd.DataFrame(x_scaled, columns=self.x.columns)

        return x_scaled_df, self.y  # return the scaled features and target variable


    def select_important_features(self, alpha=0.2, threshold_percentage=0.03):
        """
        Select important features based on Lasso regression coefficients.

        Parameters:
        - x (pd.DataFrame): The input features.
        - y (pd.Series): The target variable.
        - alpha (float): Regularization strength for Lasso.
        - threshold_percentage (float): Threshold as a percentage of the max absolute coefficient.

        Returns:
        - x_important (pd.DataFrame): DataFrame with only the important features.
        - important_features (pd.Series): Series of important feature coefficients.
        """
        # Standardize the features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(self.x)

        # Fit Lasso model
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(x_scaled, self.y)

        # Get the coefficients and identify important features
        coefficients = pd.Series(lasso.coef_, index=self.x.columns)
        max_abs_coef = abs(coefficients).max()

        # Define a threshold for "close to zero" based on max coefficient
        threshold = max_abs_coef * threshold_percentage
        important_features = coefficients[abs(coefficients) > threshold].sort_values(ascending=False)

        # Create a new dataframe with only the important features
        x_important = self.x[important_features.index]

        return x_important, important_features