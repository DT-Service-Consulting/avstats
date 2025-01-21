# core/ML/Multicollinearity.py
import pandas as pd
from pandas import DataFrame
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Union, Any, Tuple
from avstats.core.EDA.validators.validator_Multicollinearity import MulticollinearityInput


class Multicollinearity:
    def __init__(self, scaled_df: pd.DataFrame, y: pd.Series) -> None:
        """
        Initialize the Multicollinearity class.

        Parameters:
        scaled_df (pd.DataFrame): The standardized DataFrame containing features.
        y (pd.Series): The target variable.
        verbose (bool): Whether to print details during processing.
        """
        # Validate inputs using Pydantic
        validated_inputs = MulticollinearityInput(scaled_df=scaled_df, y=y)

        # Store validated inputs
        self.scaled_df = validated_inputs.scaled_df
        self.y = validated_inputs.y

    def remove_high_vif_features(self, threshold: Union[int, float] = 10) -> tuple[
        DataFrame, DataFrame | Any, DataFrame]:
        """
        Iteratively remove features with high VIF values until all remaining features have VIF below a threshold.

        Parameters:
        threshold (float): The VIF threshold above which features will be removed (default is 10).

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - A DataFrame with the target variable and features with VIF below the threshold.
            - The final features DataFrame.
            - The VIF DataFrame with all remaining features.
        """
        features = self.scaled_df.copy()
        removed_features = []

        # Remove constant features before calculating VIF
        constant_features = [col for col in features.columns if features[col].nunique() <= 1]
        if constant_features:
            features = features.drop(columns=constant_features)
            removed_features.extend(constant_features)

        while True:
            # Calculate VIF for each feature
            vif_data = pd.DataFrame({
                "feature": features.columns,
                "VIF": [
                    variance_inflation_factor(features.values, i)
                    for i in range(features.shape[1])
                ]
            }).round(2)

            # Remove infinite or NaN VIF values
            vif_data = vif_data.replace([float('inf'), float('nan')], float('inf'))
            infinite_vif_features = vif_data[vif_data['VIF'] == float('inf')]['feature']
            if not infinite_vif_features.empty:
                features = features.drop(columns=infinite_vif_features)
                removed_features.extend(list(infinite_vif_features))
                continue

            # Check if all features meet the VIF threshold
            if vif_data.empty or (vif_data['VIF'] <= threshold).all():
                break

            # Remove the feature with the highest VIF value
            feature_to_remove = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
            removed_features.append(feature_to_remove)
            features = features.drop(columns=feature_to_remove)

        return pd.concat([self.y, features], axis=1), features, vif_data