# core/ML/ModelTraining.py
import seaborn as sns
import xgboost as xgb
from typing import Tuple
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.regression.linear_model import RegressionResults
from avstats.core.ML.ModelEvaluation import *
from avstats.core.ML.validators.validator_ModelTraining import ModelTrainingInput


class ModelTraining:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Initializes the ModelTraining class with training and testing data.

        Parameters:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        x_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target values.
        """
        # Validate inputs using Pydantic
        validated_inputs = ModelTrainingInput(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

        # Store validated inputs
        self.x_train = validated_inputs.x_train
        self.y_train = validated_inputs.y_train
        self.x_test = validated_inputs.x_test
        self.y_test = validated_inputs.y_test
        self.model = None
        self.y_pred = None

        # Directory with available models
        self.models = {
            "Linear Regression": self.train_linear_model,
            "Decision Tree": self.train_decision_tree,
            "Random Forest": self.train_random_forest,
            "SVR": self.train_svr,
            "Gradient Boosting": self.train_gbm,
            "XGBoost": self.train_xgboost
        }

    def train_linear_model(self) -> tuple[RegressionResults, np.ndarray]:
        """
        Trains a Linear Regression model using statsmodels and provides a detailed summary.

        Returns:
        Tuple[sm.OLS, np.ndarray]: Trained OLS Regression model and predicted values for x_test.
        """
        # Check if a constant column already exists
        x_train_with_const = sm.add_constant(self.x_train, has_constant='add')
        x_test_with_const = sm.add_constant(self.x_test, has_constant='add')

        # Fit the model using statsmodels OLS
        self.model = sm.OLS(self.y_train, x_train_with_const).fit()
        self.y_pred = self.model.predict(x_test_with_const)

        return self.model, self.y_pred

    def train_decision_tree(self) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        """
        Trains a Decision Tree Regressor using the training data.

        Returns:
        Tuple[DecisionTreeRegressor, np.ndarray]: Trained Decision Tree model and predicted values for x_test.
        """
        # Initialize the Decision Tree Regressor
        self.model = DecisionTreeRegressor(max_depth=5, random_state=42)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)

        return self.model, self.y_pred

    def train_random_forest(self) -> Tuple[RandomForestRegressor, np.ndarray]:
        """
        Trains a Random Forest Regressor using the training data.

        Returns:
        Tuple[RandomForestRegressor, np.ndarray]: Trained Random Forest Regressor model and predicted values for x_test.
        """
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)

        return self.model, self.y_pred

    def train_svr(self) -> Tuple[SVR, np.ndarray]:
        """ Trains Support Vector Regression (SVR) model. """
        self.model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)

        return self.model, self.y_pred

    def train_gbm(self) -> Tuple[GradientBoostingRegressor, np.ndarray]:
        """ Trains a Gradient Boosting Machine (GBM) model. """
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)

        return self.model, self.y_pred

    def train_xgboost(self) -> Tuple[xgb.XGBRegressor, np.ndarray]:
        """ Trains an XGBoost model. """
        self.model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)

        return self.model, self.y_pred

    def plot_model(self, title, evaluation_metrics, ax=None) -> None:
        """
        Plots actual vs predicted values with a metrics table.

        Parameters:
        - title (str): Title for the plot.
        - evaluation_metrics (dict): Evaluation metrics to display in the plot.
        - ax (matplotlib.axes._subplots.AxesSubplot): Axis for the subplot (optional).
        """
        if self.y_pred is None:
            raise ValueError("You need to call train_linear_model() before plotting.")

        sns.set_theme(style="whitegrid")

        # If no axis is provided, create a new plot
        if ax is None:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()

        # Plot scatter and perfect prediction line
        sns.scatterplot(x=self.y_test, y=self.y_pred, ax=ax, alpha=0.7, label='Predictions')
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', lw=2,
                label='Perfect Prediction', )

        # Add details to the plot
        ax.set_title(f"{title} - Actual vs Predicted", pad=20)
        ax.set_xlabel("Actual Values (min.)")
        ax.set_ylabel("Predicted Values (min.)")
        ax.set_xlim(self.y_test.min(), self.y_test.max())
        ax.set_ylim(self.y_test.min(), self.y_test.max())
        ax.legend()
        metrics_box(evaluation_metrics, ax)
