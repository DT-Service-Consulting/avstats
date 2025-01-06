import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelComparison:
    """
    A class for evaluating and visualizing model performance.
    """
    def __init__(self, actual, predicted, residuals=None):
        """
        Initialize ModelEvaluation with actual, predicted, and residual values.

        Args:
            actual (array-like): The actual values.
            predicted (array-like): The predicted values.
            residuals (array-like, optional): Residuals (actual - predicted). Defaults to None.
        """
        self.actual = np.array(actual)
        self.predicted = np.array(predicted)
        self.residuals = np.array(residuals) if residuals is not None else self.actual - self.predicted

    def evaluate_metrics(self):
        """
        Calculate evaluation metrics: MAE, MAPE, and RMSE.

        Returns:
            dict: A dictionary containing MAE, MAPE, and RMSE.
        """
        mae = mean_absolute_error(self.actual, self.predicted)
        mape = np.mean(np.abs(self.residuals / self.actual)) * 100 if np.all(self.actual != 0) else np.nan
        rmse = np.sqrt(mean_squared_error(self.actual, self.predicted))
        return {"MAE (min.)": mae, "MAPE (%)": mape, "RMSE (min.)": rmse}

    def plot_combined(self, model_name):
        """
        Plot actual vs predicted values and residuals side by side.

        Args:
            model_name (str): Name of the model.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Actual vs Predicted plot
        axes[0].plot(self.actual, label='Actual', alpha=0.7)
        axes[0].plot(self.predicted, label='Predicted', color='orange')
        axes[0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0].legend()
        axes[0].tick_params(axis='x', labelsize=8)

        # Residuals plot
        axes[1].plot(self.residuals, label='Residuals', color='purple')
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.7)
        axes[1].set_title(f'{model_name}: Residuals')
        axes[1].legend()
        axes[1].tick_params(axis='x', labelsize=8)

        plt.tight_layout()
        plt.show()
