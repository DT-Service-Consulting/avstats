# core/ML_workflow/ModelEvaluation.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from typing import Optional, Tuple
from avstats.core.ML_workflow.validators_ML.validator_model_evaluation import CrossValidationInput, ModelEvaluationInput


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


def evaluate_model(test_data: np.ndarray, predictions: np.ndarray, residuals: Optional[np.ndarray] = None
                   ) -> Tuple[float, Optional[float], float]:
    """
    Evaluate model performance using MAE, MAPE, and RMSE.

    Parameters:
    test_data (np.ndarray): Actual target values for testing.
    predictions (np.ndarray): Predicted values from the model.
    residuals (Optional[np.ndarray]): Difference between actual and predicted values. Default is None.

    Returns:
    Tuple[float, Optional[float], float]:
        - Mean Absolute Error (MAE)
        - Mean Absolute Percentage Error (MAPE) (if residuals are provided)
        - Root Mean Squared Error (RMSE)
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
    mape = np.mean(abs(residuals / test_data)) * 100 if residuals is not None else None
    rmse = root_mean_squared_error(test_data, predictions)

    print(f'Mean Absolute Error (MAE): {mae:.2f}min.')
    print(f'Mean Absolute Percent Error (MAPE): {mape:.2f}%' if mape is not None else "MAPE not available")
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}min.')
    return mae, mape, rmse
