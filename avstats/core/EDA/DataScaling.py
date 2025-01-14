# core/ML/OneHotEncoding.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from avstats.core.EDA.validators.validator_DataScaling import DataPreparationInput


class DataScaling:
    def __init__(self, df: pd.DataFrame, target_variable: str):
        """
        Initialize the DataPreparation object.

        Args:
            df (pd.DataFrame): The input data.
            target_variable (str): The name of the target variable column.
        """
        # Validate inputs using Pydantic
        validated_inputs = DataPreparationInput(df=df, target_variable=target_variable)

        # Store validated input
        self.df = validated_inputs.df
        self.target_variable = validated_inputs.target_variable # total_dep_delay with routes, dep_delay with weather
        self.x = None
        self.y = None

    def standardize_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Standardize the input features by removing the mean and scaling to unit variance.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing the standardized features
            (as a DataFrame) and the target variable (as a Series).
        """
        # Prepare data
        self.x = self.df.drop(columns=[self.target_variable])  # drop the target variable
        self.y = self.df[self.target_variable]

        # Create the scaler instance
        scaler = StandardScaler()

        # Apply scaling to the features
        x_scaled = scaler.fit_transform(self.x)

        # Convert the scaled data back into a DataFrame for easier handling
        x_scaled_df = pd.DataFrame(x_scaled, columns=self.x.columns)

        return x_scaled_df, self.y  # return the scaled features and target variable


    def select_important_features(self, alpha: float = 0.2, threshold_percentage: float = 0.03
                                  ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Select important features based on Lasso regression coefficients.

        Args:
            alpha (float): Regularization strength for Lasso. Defaults to 0.2.
            threshold_percentage (float): Threshold as a percentage of the max absolute coefficient.
                                           Defaults to 0.03.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - x_important: A DataFrame with only the important features.
                - important_features: A Series of important feature coefficients.
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