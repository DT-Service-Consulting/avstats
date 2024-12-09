# core/ML_workflow/ModelEvaluation.py
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import numpy as np
import statsmodels.api as sm


def cross_validate(x_train, y_train, cv=5):
    """
    Perform k-fold cross-validation on the model.

    Parameters:
    cv (int): Number of cross-validation folds. Default is 5.

    Returns:
    array: Cross-validation R2 scores.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(x_train):
        x_train_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Fit a new OLS model for each fold
        ols_model = sm.OLS(y_train_fold, sm.add_constant(x_train_fold))
        model_fit = ols_model.fit()
        y_pred_fold = model_fit.predict(sm.add_constant(x_val_fold))

        scores.append(r2_score(y_val_fold, y_pred_fold))

    return np.array(scores)


def evaluate_model(test_data, predictions, residuals=None):
    """Evaluate model performance using MAE, MAPE, and RMSE. (y_test & y_pred)
    """
    mae = mean_absolute_error(test_data, predictions)
    mape = np.mean(abs(residuals / test_data)) * 100 if residuals is not None else None
    rmse = root_mean_squared_error(test_data, predictions)

    print(f'Mean Absolute Error (MAE): {mae:.2f}min.')
    print(f'Mean Absolute Percent Error (MAPE): {mape:.2f}%' if mape is not None else "MAPE not available")
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}min.')
    return mae, mape, rmse

