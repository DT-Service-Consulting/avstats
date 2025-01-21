# ColumnsCorrelation.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency


class ColumnsCorrelation:
    def __init__(self, df):

        self.df = df

    @staticmethod
    def cramers_v(x, y):
        """
        Compute Cramér's V for two categorical variables.
        """
        contingency_table = pd.crosstab(x, y)
        chi2 = chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        return np.sqrt(phi2 / min(r - 1, k - 1))

    def categorical_correlation_matrix(self, categorical_columns):
        """
        Compute pairwise Cramér's V correlations for categorical columns.
        """
        corr_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
        for col1 in categorical_columns:
            for col2 in categorical_columns:
                if col1 == col2:
                    corr_matrix.loc[col1, col2] = 1.0
                else:
                    try:
                        corr_matrix.loc[col1, col2] = self.cramers_v(self.df[col1], self.df[col2])
                    except:
                        corr_matrix.loc[col1, col2] = np.nan # Handle cases where computation fails
        return corr_matrix.astype(float)

    def plot_correlation_heatmaps(self):
        """
        Generates two heatmaps:
        1. Correlation heatmap for numerical columns.
        2. Correlation heatmap for categorical columns.

        Parameters:
        - df (pd.DataFrame): The dataframe with both numerical and categorical columns.
        """
        # Separate numerical and categorical columns
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

        # Compute correlation matrices
        numerical_corr = self.df[numerical_columns].corr()
        categorical_corr =self.categorical_correlation_matrix(categorical_columns)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(25, 10)) # Adjusted figure size for better readability

        # Plot heatmap for numerical correlations
        sns.heatmap(numerical_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0], cbar_kws={"shrink": 0.7})
        axes[0].set_title("Correlation Heatmap for Numerical Columns", fontsize=14)
        axes[0].tick_params(axis='x', rotation=45, labelsize=10) # Increased font size
        axes[0].tick_params(axis='y', labelsize=10)

        # Plot heatmap for categorical correlations
        sns.heatmap(categorical_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1], cbar_kws={"shrink": 0.7})
        axes[1].set_title("Correlation Heatmap for Categorical Columns", fontsize=14)
        axes[1].tick_params(axis='x', rotation=45, labelsize=10) # Increased font size
        axes[1].tick_params(axis='y', labelsize=10)

        plt.tight_layout()
        plt.show()
