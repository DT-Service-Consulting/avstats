# core/ML_workflow/ModelTraining.py
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class ModelTraining:
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initializes the ModelTraining class with training and testing data.

        Parameters:
        X (array-like): Features data.
        y (array-like): Target data.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None
        self.y_pred = None

    def train_linear_model(self):
        """
        Trains a Linear Regression model using statsmodels and provides a detailed summary.

        Returns:
        statsmodels.OLS: Trained OLS Regression model.
        array-like: Predicted values for x_test.
        """
        # Add a constant (intercept) to the model
        x_train_with_const = sm.add_constant(self.x_train)
        x_test_with_const = sm.add_constant(self.x_test)

        # Fit the model using statsmodels OLS
        self.model = sm.OLS(self.y_train, x_train_with_const).fit()
        self.y_pred = self.model.predict(x_test_with_const)

        return self.model, self.y_pred

    def train_random_forest(self):
        """
        Trains a Random Forest Regressor using the training data.

        Returns:
        RandomForestRegressor: Trained Random Forest Regressor model.
        array-like: Predicted values for x_test.
        """
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)

        return self.model, self.y_pred

    def plot_model(self):
        """
        Plots the model's predicted values against the actual values using a scatter plot.
        """
        if self.y_pred is None:
            raise ValueError("You need to call predict() before plotting.")

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.7)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', lw=2)  # Diagonal line
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.xlim(self.y_test.min(), self.y_test.max())
        plt.ylim(self.y_test.min(), self.y_test.max())
        plt.grid()
        plt.show()

    def tune_and_evaluate(self, param_grid, verbose, search_type='grid', cv=5, scoring='neg_mean_squared_error', n_iter=10):
        """
        Performs hyperparameter tuning (Grid Search or Randomized Search) and evaluates the best model on test data.

        Parameters:
        model: The model instance (e.g., RandomForestRegressor()) to be tuned and evaluated.
        param_grid: Dictionary containing parameter grid for hyperparameter tuning.
        search_type: Type of search ('grid' for GridSearchCV, 'random' for RandomizedSearchCV).
        cv: Number of cross-validation folds.
        scoring: Scoring metric for evaluation.
        n_iter: Number of parameter settings sampled in RandomizedSearchCV.

        Returns:
        dict: The best parameters and evaluation metrics.
        """
        if search_type == 'grid':
            search = GridSearchCV(estimator=RandomForestRegressor(n_estimators=200, random_state=42),
                                  param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=verbose)
        elif search_type == 'random':
            search = RandomizedSearchCV(estimator=RandomForestRegressor(n_estimators=200, random_state=42),
                                        param_distributions=param_grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, verbose=verbose)
        else:
            raise ValueError("Invalid search_type. Choose 'grid' or 'random'.")

        # Fit search on the training data
        search.fit(self.x_train, self.y_train)

        # Retrieve best model
        best_model = search.best_estimator_
        best_model.fit(self.x_train, self.y_train)
        best_parameters = search.best_params_
        self.y_pred = best_model.predict(self.x_test)

        return best_model, best_parameters, self.y_pred