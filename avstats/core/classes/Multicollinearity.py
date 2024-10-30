import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Union


class Multicollinearity:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def calculate_vif(self) -> pd.DataFrame:
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

    def remove_high_vif_features(self, target_variable: str, threshold: Union[int, float] = 15) -> pd.DataFrame:
        """
        Iteratively remove features with high VIF values until all remaining features have VIF below a threshold.

        Parameters:
        target_variable (str): The target variable to exclude from VIF calculation.
        threshold (float): The VIF threshold above which features will be removed (default is 10).

        Returns:
        pd.DataFrame: A DataFrame with the target variable and features with VIF below the threshold.
        """
        features = self.df.drop(columns=target_variable).copy()

        while True:
            vif_data = pd.DataFrame({
                "feature": features.columns,
                "VIF": [
                    variance_inflation_factor(features.values, i)
                    if features.iloc[:, i].var() != 0 else float('inf')
                    for i in range(features.shape[1])
                ]
            })

            # Debug: print the VIF values
            print("VIF Data:\n", vif_data)

            # Remove features with infinite VIF
            infinite_vif_features = vif_data[vif_data['VIF'] == float('inf')]['feature']
            if not infinite_vif_features.empty:
                print(f"Removing features with infinite VIF: {list(infinite_vif_features)}")
                features = features.drop(columns=infinite_vif_features)

            # Drop rows in VIF DataFrame where VIF is NaN or infinite
            vif_data = vif_data.dropna().replace([float('inf')], pd.NA).dropna()

            if vif_data.empty or (vif_data['VIF'] <= threshold).all():
                print("No features above the VIF threshold or no features left.")
                break

            # Remove the feature with the highest VIF value
            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            print(f"Removing feature: {feature_to_remove} with VIF: {vif_data['VIF'].max()}")
            features = features.drop(columns=feature_to_remove)

        return pd.concat([self.df[target_variable], features], axis=1)
