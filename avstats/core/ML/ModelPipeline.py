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

    def rolling_forecast(self, train_window=30, forecast_horizon=1):
        """
        Perform rolling forecast evaluation for time-series models.
        Args:
            train_window (int): Number of time steps in the rolling training window.
            forecast_horizon (int): Number of steps to forecast at each iteration.
        Returns:
            dict: Evaluation metrics (MAE and RMSE) and forecasts.
        """
        if self.model_type not in ["arima", "sarimax"]:
            raise ValueError("Rolling forecast is applicable only to ARIMA/SARIMAX models.")

        data = self.df['total_dep_delay'].astype(float)
        n = len(data)
        predictions = []
        actual_values = []

        # Default parameter handling
        default_params = {"order": (1, 1, 1)}  # Default ARIMA params
        params = self.results.get("best_params", default_params)

        for start in range(n - train_window - forecast_horizon + 1):
            # Define rolling train and test sets
            train_data = data[start: start + train_window]
            test_data = data[start + train_window: start + train_window + forecast_horizon]

            # Skip if insufficient data for a window
            if len(train_data) == 0 or len(test_data) == 0:
                logging.warning(f"Skipping rolling window at start={start} due to insufficient data.")
                continue

            # Fit the model
            try:
                if self.model_type == "arima":
                    model = self.model(endog=train_data, **params).fit()
                elif self.model_type == "sarimax":
                    model = self.model(endog=train_data, exog=None, **params).fit()

                # Forecast
                forecast = model.forecast(steps=forecast_horizon)
                predictions.extend(forecast)
                actual_values.extend(test_data)
            except Exception as e:
                logging.warning(f"Error during rolling forecast: {e}")
                continue

        # Calculate metrics
        if len(predictions) > 0 and len(actual_values) > 0:
            mae = mean_absolute_error(actual_values, predictions)
            rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        else:
            mae, rmse = float("inf"), float("inf")  # Handle cases with no valid forecasts

        return {
            "mae": mae,
            "rmse": rmse,
            "predictions": predictions,
            "actual_values": actual_values,
        }