# core/ML/Multicollinearity.py
import pandas as pd
from pandas import DataFrame
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Union, Any
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

        # Identifying constant features correctly
        constant_features = features.nunique(dropna=True)
        constant_features = constant_features[constant_features <= 1].index.tolist()

        # Drop constant features from the dataset
        if constant_features:
            features = features.drop(columns=constant_features)
            removed_features.extend(constant_features)

        while True:
            # Calculate VIF for each feature
            vif_data = pd.DataFrame({
                "feature": features.columns,
                "VIF": [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
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

            self.y = self.y.reset_index(drop=True)
            features = features.reset_index(drop=True)

        return pd.concat([self.y, features], axis=1), features, vif_data

    def pearson_correlation(self, threshold=0.5):
        """
        Computes Pearson correlation between features and the target variable.
        Removes features that have a high correlation (greater than the threshold).

        Parameters:
        df (pd.DataFrame): The input dataframe.
        target_variable (pd.Series): The target variable.
        threshold (float): Correlation threshold above which features will be removed.

        Returns:
        pd.DataFrame: Filtered dataframe with low correlation features.
        pd.DataFrame: Dataframe displaying correlation values.
        """
        # Avoid modifying the original DataFrame and Reset index to align data properly
        self.scaled_df = self.scaled_df.reset_index(drop=True)
        target_variable = self.y.reset_index(drop=True)

        # Ensure target_variable is a Series and align indices
        if isinstance(target_variable, pd.Series):
            target_variable = target_variable.reindex(self.scaled_df.index)  # Align index
            self.scaled_df[target_variable.name] = target_variable
        else:
            raise ValueError("target_variable must be a pandas Series with the same index as df")

        # Check if target exists in df
        target_name = target_variable.name
        if target_name not in self.scaled_df.columns:
            raise KeyError(f"Target variable '{target_name}' not found in DataFrame columns")

        # Compute correlation matrix
        correlation_matrix = self.scaled_df.corr(method='pearson')

        # Extract correlation with the target variable
        target_correlation = correlation_matrix[[target_name]].drop(index=target_name).rename(
            columns={target_name: 'Correlation'})

        # Identify features with absolute correlation above the threshold
        features_to_remove = target_correlation[abs(target_correlation['Correlation']) > threshold].index.tolist()

        # Adjust threshold if too few features remain
        while len(self.scaled_df.columns) - len(features_to_remove) < 5 and threshold < 0.9:
            threshold += 0.05
            features_to_remove = target_correlation[abs(target_correlation['Correlation']) > threshold].index.tolist()

        # Remove highly correlated features
        df_filtered = self.scaled_df.drop(columns=features_to_remove, errors='ignore')

        return df_filtered, target_correlation