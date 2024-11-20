# core/ML_workflow/Multicollinearity.py
import pandas as pd
from pandas import DataFrame
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Union, Any

class Multicollinearity:
    def __init__(self, scaled_df: pd.DataFrame, y: pd.Series, verbose: bool = True) -> None:
        """
        Initialize the Multicollinearity class.

        Parameters:
        scaled_df (pd.DataFrame): The standardized DataFrame containing features.
        y (pd.Series): The target variable.
        verbose (bool): Whether to print details during processing.
        """
        self.scaled_df = scaled_df
        self.y = y
        self.verbose = verbose

    def remove_high_vif_features(self, threshold: Union[int, float] = 10) -> tuple[DataFrame, Union[DataFrame, Any]]:
        """
        Iteratively remove features with high VIF values until all remaining features have VIF below a threshold.

        Parameters:
        threshold (float): The VIF threshold above which features will be removed (default is 10).

        Returns:
        pd.DataFrame: A DataFrame with the target variable and features with VIF below the threshold.
        """
        features = self.scaled_df.copy()
        removed_features = []

        # Remove constant features before calculating VIF
        constant_features = [col for col in features.columns if features[col].nunique() == 1]
        if constant_features:
            if self.verbose:
                print(f"Removing constant features: {constant_features}")
            features = features.drop(columns=constant_features)
            removed_features.extend(constant_features)

        while True:
            vif_data = pd.DataFrame({
                "feature": features.columns,
                "VIF": [
                    variance_inflation_factor(features.values, i)
                    if features.iloc[:, i].var() != 0 and not features.iloc[:, i].isnull().all() else float('inf')
                    for i in range(features.shape[1])
                ]
            }).round(2)

            # Debug: Print VIF values if verbose mode is on
            if self.verbose:
                print("Current VIF Data:\n", vif_data)

            # Remove features with infinite VIF
            infinite_vif_features = vif_data[vif_data['VIF'] == float('inf')]['feature']
            if not infinite_vif_features.empty:
                if self.verbose:
                    print(f"Removing features with infinite VIF: {list(infinite_vif_features)}")
                features = features.drop(columns=infinite_vif_features)
                removed_features.extend(list(infinite_vif_features))
                continue

            # Check if all features meet the VIF threshold
            if vif_data.empty or (vif_data['VIF'] <= threshold).all():
                print("All remaining features have VIF below the threshold.")
                print("Final VIF values after feature removal:\n", vif_data)
                break

                # Remove the feature with the highest VIF value
            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            removed_features.append(feature_to_remove)
            if self.verbose:
                print(f"Removing feature: {feature_to_remove} with VIF: {vif_data['VIF'].max()}")
            features = features.drop(columns=feature_to_remove)

        if self.verbose and removed_features:
            print("Summary of removed features:", removed_features)

        # Combine the remaining features with the target variable and return
        return pd.concat([self.y, features], axis=1), features