# core/ML_pipelines/ModelEvaluation.py
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

class ModelEvaluation:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = model.predict(x_test)

    def evaluate_model(self, model):
        """
        Evaluates the given model on the testing data.

        Parameters:
        model (object): The model to evaluate.

        Returns:
        dict: Dictionary containing MAE, MSE, and R2 score.
        """
        test_mae = mean_absolute_error(self.y_test, self.y_pred)
        test_mse = mean_squared_error(self.y_test, self.y_pred)
        test_r2 = r2_score(self.y_test, self.y_pred)

        return {'MAE': f"{test_mae:.2f}", 'MSE': f"{test_mse:.2f}", 'R2': f"{test_r2:.2f}"}

    def cross_validate(self, model, cv=5):
        """
        Performs cross-validation on the given model using the training data.

        Parameters:
        model (object): The model to cross-validate.
        cv (int): Number of cross-validation folds. Default is 5.

        Returns:
        array: Cross-validation scores.
        """
        cv_metrics = cross_val_score(model, self.x_train, self.y_train, cv=cv)
        print(f"Cross-validation Metrics: {cv_metrics}")

        return cv_metrics