import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


class Multicollinearity:
    def __init__(self, dataframe):
        self.df = dataframe

    def calculate_vif(self):
        """
        Calculate Variance Inflation Factor (VIF) for each feature to assess multicollinearity.

        Returns:
        pd.DataFrame: A DataFrame with features and their corresponding VIF values.
        """
        vif_data = pd.DataFrame({
            "feature": self.df.columns,
            "VIF": [variance_inflation_factor(self.df.values, i) for i in range(self.df.shape[1])]
        })
        return vif_data

    def remove_high_vif_features(self, target_variable, threshold=10):
        """
        Iteratively remove features with high VIF values until all remaining features have VIF below a threshold.

        Parameters:
        target_variable (str): The target variable to exclude from VIF calculation.
        threshold (float): The VIF threshold above which features will be removed (default is 10).

        Returns:
        pd.DataFrame: A DataFrame with the target variable and features with VIF below the threshold.
        """
        features = self.df.drop(columns=target_variable)

        while True:
            vif_data = pd.DataFrame({
                "feature": features.columns,
                "VIF": [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
            })

            if (vif_data['VIF'] <= threshold).all():
                print("No features above the VIF threshold.")
                break

            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            print(f"Removing feature: {feature_to_remove}")
            features = features.drop(columns=feature_to_remove)

        return pd.concat([self.df[target_variable], features], axis=1)
