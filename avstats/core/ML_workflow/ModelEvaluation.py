# core/ML_workflow/ModelEvaluation.py
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import numpy as np
import statsmodels.api as sm

class ModelEvaluation:
    def __init__(self, model, y_pred, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = y_pred

    def evaluate_model(self):
        """
        Evaluates the given model on the testing data.

        Parameters:
        model (object): The model to evaluate.

        Returns:
        dict: Dictionary containing MAE, MSE, and R2 score.
        """
        test_mae = mean_absolute_error(self.y_test, self.y_pred)
        test_mse = root_mean_squared_error(self.y_test, self.y_pred)
        test_r2 = r2_score(self.y_test, self.y_pred)

        return {'MAE': f"{test_mae:.2f}", 'MSE': f"{test_mse:.2f}", 'R2': f"{test_r2:.2f}"}

    def cross_validate(self, cv=5):
        """
        Perform k-fold cross-validation on the model.

        Parameters:
        cv (int): Number of cross-validation folds. Default is 5.

        Returns:
        array: Cross-validation R2 scores.
        """
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []

        for train_index, val_index in kf.split(self.x_train):
            x_train_fold, x_val_fold = self.x_train.iloc[train_index], self.x_train.iloc[val_index]
            y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            # Fit a new OLS model for each fold
            ols_model = sm.OLS(y_train_fold, sm.add_constant(x_train_fold))
            model_fit = ols_model.fit()
            y_pred_fold = model_fit.predict(sm.add_constant(x_val_fold))

            scores.append(r2_score(y_val_fold, y_pred_fold))

        return np.array(scores)


class ComplexModelEvaluation:
    def __init__(self, test_data, predictions, residuals=None):
        self.test_data = test_data
        self.predictions = predictions
        self.residuals = residuals

    def evaluate_model(self):
        """Evaluate model performance using MAE, MAPE, and RMSE."""
        mae = mean_absolute_error(self.test_data, self.predictions)
        mape = np.mean(abs(self.residuals / self.test_data)) * 100 if self.residuals is not None else None
        rmse = root_mean_squared_error(self.test_data, self.predictions)

        print(f'Mean Absolute Error (MAE): {mae:.2f}min.')
        print(f'Mean Absolute Percent Error (MAPE): {mape:.2f}%' if mape is not None else "MAPE not available")
        print(f'Root Mean Squared Error (RMSE): {rmse:.2f}min.')
        return mae, mape, rmse

