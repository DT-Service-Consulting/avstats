import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


class ModelPipeline:
    def __init__(
            self,
            model: Any,
            df: Optional[pd.DataFrame] = None,
            param_grid: Optional[Union[Dict, List[Dict]]] = None,
            model_type: str = "ml",
            scoring: str = "neg_mean_squared_error",
    ):
        self.model = model
        self.df = df
        self.param_grid = param_grid
        self.model_type = model_type
        self.scoring = scoring
        self.best_model = None
        self.results = {}

    def hyperparameter_tuning(self, x_train=None, y_train=None):
        """
        Perform hyperparameter tuning for both ML and time-series models.
        """
        if not self.param_grid:
            logging.warning(f"No hyperparameter grid provided for {self.model}. Skipping tuning.")
            if self.model_type not in ["arima", "sarimax"]:
                self.best_model = self.model.fit(x_train, y_train)
            self.results["best_params"] = "No tuning performed"
            return self.best_model

        if self.model_type in ["arima", "sarimax"]:
            best_score = float("inf")
            for params in self.param_grid:
                try:
                    # Ensure endog is a pd.Series
                    endog = self.df['total_dep_delay'].astype(float)

                    if self.model_type == "arima":
                        model = self.model(endog=endog, **params).fit()
                    elif self.model_type == "sarimax":
                        model = self.model(endog=endog, exog=None, **params).fit()

                    # Evaluate model
                    test_data = endog[-30:]  # Use the last 30 days as test data
                    mae = self.evaluate_arima(model, test_data=test_data)

                    # Track best parameters
                    if mae < best_score:
                        best_score = mae
                        self.best_model = model
                        self.results["best_params"] = params
                except Exception as e:
                    logging.warning(f"Error with parameters {params}: {e}")
        else:
            # ML models
            search = GridSearchCV(self.model, self.param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
            search.fit(x_train, y_train)
            self.best_model = search.best_estimator_
            self.results["best_params"] = search.best_params_
        logging.info(f"Best Parameters: {self.results.get('best_params', 'No valid parameters')}")
        return self.best_model

    @staticmethod
    def evaluate_arima(model, test_data):
        """
        Evaluate ARIMA/SARIMAX using Mean Absolute Error (MAE).
        """
        try:
            forecast = model.forecast(steps=len(test_data))
            mae = mean_absolute_error(test_data, forecast)
            return mae
        except Exception as e:
            logging.error(f"Error during ARIMA evaluation: {e}")
            return float("inf")

    def plot_learning_curve(self, x, y, title, cv=5):
        """
        Plot the learning curve for ML models.
        """
        if self.model_type in ["arima", "sarimax"]:
            logging.info(f"Learning curves are not applicable for {self.model_type} models.")
            return

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, x, y, cv=cv, scoring=self.scoring, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_mean = -np.mean(train_scores, axis=1)
        test_mean = -np.mean(test_scores, axis=1)

        plt.figure()
        plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
        plt.plot(train_sizes, test_mean, label="Validation Score", color="orange")
        plt.title(f"Learning Curve: {title}")
        plt.xlabel("Training Size")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

    def train_and_evaluate(self, x_train, x_test, y_train, y_test):
        """
        Train and evaluate the model.
        """
        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        self.results["metrics"] = {"mae": mae, "rmse": rmse}
        logging.info(f"Evaluation Metrics: MAE={mae}, RMSE={rmse}")
        return self.model, predictions
