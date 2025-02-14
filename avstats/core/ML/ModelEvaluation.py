# core/ML/ModelEvaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from typing import List, Dict, Union, Optional
from avstats.core.ML.validators.validator_ModelEvaluation import ModelEvaluationInput


def cross_validate(x_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> np.ndarray:
    """
    Perform k-fold cross-validation on the model.

    Parameters:
    x_train (np.ndarray): Features for training the model.
    y_train (np.ndarray): Target variable for training.
    cv (int): Number of cross-validation folds. Default is 5.

    Returns:
    np.ndarray: Cross-validation R2 scores for each fold.
    """
    # Ensure x_train and y_train are numpy arrays
    if isinstance(x_train, pd.DataFrame):
        x_train = x_train.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()

    # Validate inputs
    assert not np.any(np.isnan(x_train)), "x_train contains NaN values."
    assert not np.any(np.isnan(y_train)), "y_train contains NaN values."
    assert np.all(np.isfinite(x_train)), "x_train contains infinite values."
    assert np.all(np.isfinite(y_train)), "y_train contains infinite values."

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(x_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Add constant to both training and validation sets
        x_train_fold_with_const = sm.add_constant(x_train_fold, has_constant='add')
        x_val_fold_with_const = sm.add_constant(x_val_fold, has_constant='add')

        # Fit OLS model
        ols_model = sm.OLS(y_train_fold, x_train_fold_with_const)
        model_fit = ols_model.fit()
        y_pred_fold = model_fit.predict(x_val_fold_with_const)

        # Calculate R2 score
        scores.append(r2_score(y_val_fold, y_pred_fold))

    return np.array(scores)

def evaluate_model(test_data: np.ndarray, predictions: np.ndarray, residuals: Optional[np.ndarray] = None) -> Dict[str, Union[float, None]]:
    """
    Evaluate model performance using MAE, MAPE, and RMSE.

    Parameters:
    test_data (np.ndarray): Actual target values for testing.
    predictions (np.ndarray): Predicted values from the model.
    residuals (Optional[np.ndarray]): Difference between actual and predicted values. Default is None.

    Returns:
    Dict[str, Union[float, None]]: Evaluation metrics including MAE, MAPE, and RMSE.
    """
    # Ensure all inputs are numpy arrays
    if isinstance(test_data, pd.Series):
        test_data = test_data.to_numpy()
    if isinstance(predictions, pd.Series):
        predictions = predictions.to_numpy()
    if residuals is not None and isinstance(residuals, pd.Series):
        residuals = residuals.to_numpy()

    # Validate inputs using Pydantic
    ModelEvaluationInput(test_data=test_data, predictions=predictions, residuals=residuals)

    mae = mean_absolute_error(test_data, predictions)
    rmse = root_mean_squared_error(test_data, predictions)

    # Handle zero values in test_data for MAPE calculation
    if residuals is not None:
        non_zero_mask = test_data != 0  # Exclude zero values
        mape = np.mean(abs(residuals[non_zero_mask] / test_data[non_zero_mask])) * 100 if non_zero_mask.any() else None
    else:
        mape = None

    return {
        "MAE (min.)": round(mae, 2),
        "MAPE (%)": round(mape, 2) if mape is not None else None,
        "RMSE (min.)": round(rmse, 2)
    }

def metrics_box(evaluation_metrics, ax=None):
    """
    Add a metrics box to a specific axis.

    Args:
        evaluation_metrics (dict): A dictionary of evaluation metrics.
        ax (matplotlib.axes._subplots.AxesSubplot): The subplot axis to add the metrics box to.
    """
    metrics_text = "\n\n".join([f"{key}: {value:.2f}" for key, value in evaluation_metrics.items()])
    props = dict(boxstyle="round,pad=0.4", edgecolor="gray", facecolor="whitesmoke")
    if ax is None:
        plt.text(
            1.1, 0.5, metrics_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='left', bbox=props
        )
        plt.tight_layout(rect=(0, 0, 0.8, 1))
    else:
        ax.text(
            1.05, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='left', bbox=props
        )

def plot_metrics(evaluation_results: List[Dict[str, Union[str, float, None]]]) -> None:
    """
    Plot model performance metrics (MAE, RMSE, and MAPE) as bar charts.

    Args:
        evaluation_results (List[Dict[str, Union[str, float, None]]]):
            A list of dictionaries containing model evaluation metrics.
            Each dictionary should have the following structure:
            - "Model" (str): The name of the model.
            - "MAE (min.)" (float): Mean Absolute Error in minutes.
            - "RMSE (min.)" (float): Root Mean Squared Error in minutes.
            - "MAPE (%)" (float or None): Mean Absolute Percentage Error in percentage.

    Returns:
        None: The function displays the plots directly.
    """
    df = pd.DataFrame(evaluation_results).set_index("Model")

    # Sort the metrics for MAE and RMSE in ascending order
    sorted_metrics = df[['MAE (min.)', 'RMSE (min.)']].sort_values(by=['MAE (min.)', 'RMSE (min.)'])

    # Plot MAE and RMSE
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    sorted_metrics.plot(kind='bar', ax=ax1, alpha=0.7)
    ax1.set_title('Model Performance (MAE and RMSE)', fontsize=14)
    ax1.set_ylabel('(min.)', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, horizontalalignment='center', fontsize=12)
    ax1.legend(title="Metrics")
    plt.tight_layout()
    plt.show()

    # Sort the MAPE metric in ascending order
    sorted_mape = df[['MAPE (%)']].sort_values(by='MAPE (%)')

    # Plot MAPE
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    sorted_mape.plot(kind='bar', ax=ax2, alpha=0.7)
    ax2.set_title('Model Performance (MAPE)', fontsize=14)
    ax2.set_ylabel('(%)', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_ylim(0, 100)  # Set y-axis range from 0 to 100
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, horizontalalignment='center', fontsize=12)
    ax2.legend(title="Metrics")
    for bar in ax2.patches:
        ax2.annotate(f"{bar.get_height():.2f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center',
                     va='bottom', fontsize=10, color='black')
    plt.tight_layout()
    plt.show()