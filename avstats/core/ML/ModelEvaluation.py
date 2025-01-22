# core/ML/ModelEvaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from typing import List, Dict, Union, Optional
from avstats.core.ML.validators.validator_ModelEvaluation import CrossValidationInput, ModelEvaluationInput


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

    # Validate inputs using Pydantic
    CrossValidationInput(x_train=x_train, y_train=y_train, cv=cv)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(x_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Fit a new OLS model for each fold
        ols_model = sm.OLS(y_train_fold, sm.add_constant(x_train_fold))
        model_fit = ols_model.fit()
        y_pred_fold = model_fit.predict(sm.add_constant(x_val_fold))

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

def plot_combined(model_name, actual, predicted, residuals=None):
    """
    Plot actual vs predicted values and residuals side by side.

    Args:
        model_name (str): Name of the model.
        actual (array-like): The actual values.
        predicted (array-like): The predicted values.
        residuals (array-like, optional): Residuals (actual - predicted). Defaults to None.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    residuals = np.array(residuals) if residuals is not None else actual - predicted
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Actual vs Predicted plot
    axes[0].plot(actual, label='Actual', alpha=0.7)
    axes[0].plot(predicted, label='Predicted', color='orange')
    axes[0].set_title(f'{model_name}: Actual vs Predicted')
    axes[0].legend()
    axes[0].tick_params(axis='x', labelsize=8)

    # Residuals plot
    axes[1].plot(residuals, label='Residuals', color='purple')
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.7)
    axes[1].set_title(f'{model_name}: Residuals')
    axes[1].legend()
    axes[1].tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.show()

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
    df[['MAE (min.)', 'RMSE (min.)']].plot(kind='bar', figsize=(12, 4), alpha=0.7)
    plt.title('Model Performance (MAE and RMSE)')
    plt.ylabel('(min.)')
    plt.xlabel('')
    plt.xticks(rotation=0, horizontalalignment='center')  # Rotate labels for better readability
    plt.legend(title="Metrics")
    plt.show()

    df[['MAPE (%)']].plot(kind='bar', figsize=(12, 4), alpha=0.7)
    plt.title('Model Performance (MAPE)')
    plt.ylabel('(%)')
    plt.xlabel('')
    plt.ylim(0, 100)  # Set y-axis range from 0 to 100
    plt.xticks(rotation=0, horizontalalignment='center')  # Rotate labels for better readability
    plt.legend(title="Metrics")
    plt.show()