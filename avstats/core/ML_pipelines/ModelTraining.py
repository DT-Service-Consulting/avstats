# core/ML_pipelines/ModelTraining.py
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

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
        """
        # Add a constant (intercept) to the model
        x_train_with_const = sm.add_constant(self.x_train)

        # Fit the model using statsmodels OLS
        self.model = sm.OLS(self.y_train, x_train_with_const).fit()

        return self.model

    def train_random_forest(self):
        """
        Trains a Random Forest Regressor using the training data.

        Returns:
        RandomForestRegressor: Trained Random Forest Regressor model.
        """
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.model.fit(self.x_train, self.y_train)

        return self.model

    def predict(self):
        """
        Predicts values using the trained model and stores predictions in y_pred.

        Returns:
        array-like: Predicted values for x_test.
        """
        if isinstance(self.model, sm.OLS):
            x_test_with_const = sm.add_constant(self.x_test)
            self.y_pred = self.model.predict(x_test_with_const)
        else:
            self.y_pred = self.model.predict(self.x_test)

        return self.y_pred

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
