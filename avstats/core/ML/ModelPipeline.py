import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, GridSearchCV
from typing import Callable, Dict, Tuple, Optional, Any
import logging

class ModelPipeline:
    def __init__(self, model: Any, param_grid: Optional[Dict] = None, scoring: str = 'neg_mean_squared_error'):
        """
        Initialize the ModelPipeline class.

        Args:
            model (Any): A machine learning or time series model instance.
            param_grid (Optional[Dict]): A dictionary of hyperparameters for tuning.
            scoring (str): Scoring metric for evaluation during grid search.
        """
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.best_model = None
        self.results = {}

    def hyperparameter_tuning(self, x_train: np.ndarray, y_train: np.ndarray, search_type: str = 'grid') -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

        Args:
            x_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target values.
            search_type (str): 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.

        Returns:
            Any: The best estimator after hyperparameter tuning.
        """
        if not self.param_grid:
            logging.warning("No hyperparameter grid provided. Skipping tuning.")
            return self.model

        if search_type == 'grid':
            search = GridSearchCV(self.model, self.param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        else:
            raise ValueError("Unsupported search_type. Use 'grid' for GridSearchCV.")

        search.fit(x_train, y_train)
        self.best_model = search.best_estimator_
        self.results['best_params'] = search.best_params_
        logging.info(f"Best Parameters: {search.best_params_}")
        return self.best_model

    def plot_learning_curve(self, x: np.ndarray, y: np.ndarray, title: str, cv: int = 5, n_jobs: int = -1):
        """
        Generate and plot a learning curve.

        Args:
            x (np.ndarray): Features.
            y (np.ndarray): Target values.
            title (str): Title of the plot.
            cv (int): Number of cross-validation folds.
            n_jobs (int): Number of parallel jobs for computation.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, x, y, cv=cv, scoring=self.scoring, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(-train_scores, axis=1)
        train_std = np.std(-train_scores, axis=1)
        test_mean = np.mean(-test_scores, axis=1)
        test_std = np.std(-test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
        plt.plot(train_sizes, test_mean, label="Validation Score", color="orange")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="orange", alpha=0.2)
        plt.title(f"Learning Curve: {title}")
        plt.xlabel("Training Size")
        plt.ylabel("Error")
        plt.legend(loc="best")
        plt.grid()
        plt.show()

    def train_and_evaluate(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """
        Train and evaluate the model on the given dataset.

        Args:
            x_train (np.ndarray): Training features.
            x_test (np.ndarray): Testing features.
            y_train (np.ndarray): Training target values.
            y_test (np.ndarray): Testing target values.
        """
        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)
        residuals = y_test - predictions

        # Evaluate metrics
        metrics = evaluate_model(y_test, predictions, residuals)
        self.results['metrics'] = metrics
        logging.info(f"Evaluation Metrics: {metrics}")

        # Return the trained model and predictions
        return self.model, predictions
