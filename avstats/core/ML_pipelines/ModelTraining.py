# core/ML_pipelines/ModelTraining.py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
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
        self.y_pred = None

    def train_linear_model(self):
        """
        Trains a Linear Regression model using the training data.

        Returns:
        LinearRegression: Trained Linear Regression model.
        """
        model = LinearRegression()
        model.fit(self.x_train, self.y_train)

        return model

    def train_random_forest(self):
        """
        Trains a Random Forest Regressor using the training data.

        Returns:
        RandomForestRegressor: Trained Random Forest Regressor model.
        """
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(self.x_train, self.y_train)

        return model

    def plot_model(self, model):
        """
        Plots the model's predicted values against the actual values using a scatter plot.
        """
        self.y_pred = model.predict(self.x_test)

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
