# core/ML/ModelPipeline.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from avstats.core.ML.ModelEvaluation import evaluate_model, cross_validate
from avstats.core.ML.ModelTraining import ModelTraining
from sklearn.linear_model import LinearRegression


class ModelPipeline:
    """
    A pipeline for training, evaluating, and visualizing multiple ML models on different datasets.
    """
    def __init__(self, dataframes, titles, models_to_train, param_grids=None):
        """
        Initializes the model pipeline.

        Parameters:
        - dataframes: List of tuples [(DataFrame, str)] with dataset and target column name.
        - titles: List of dataset titles.
        - models_to_train: List of model names (keys from `ModelTraining.models`).
        - param_grids: Dictionary with hyperparameter grids for each model (optional).
        """
        self.dataframes = dataframes
        self.titles = titles
        self.models_to_train = models_to_train
        self.param_grids = param_grids
        self.metrics_summary = {}
        self.final_models = {}

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # Set Seaborn style globally
        sns.set_style("whitegrid")

    def train_models(self):
        """
        Trains models for each dataset and stores results.
        """
        model_summaries = []

        for i, ((df, column), title) in enumerate(zip(self.dataframes, self.titles)):
            try:
                # Prepare features and target
                x = df.drop(columns=[column])
                y = df[column]
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,
                                                                                        random_state=42)
                model_training = ModelTraining(self.x_train, self.y_train, self.x_test, self.y_test)

                dataset_metrics = {}
                model_storage = {}  # Store all trained models + predictions per dataset

                for model_name in self.models_to_train:
                    trained_model, predictions, best_params = self.train_single_model(model_training, model_name)
                    residuals = self.y_test - predictions
                    metrics = evaluate_model(self.y_test, predictions, residuals)  # Evaluate model

                    # Store metrics
                    dataset_metrics[model_name] = {"MAE (min.)": metrics["MAE (min.)"], "MAPE (%)": metrics["MAPE (%)"],
                                                   "RMSE (min.)": metrics["RMSE (min.)"]}

                    # Choose the appropriate SHAP explainer
                    if model_name in ["Random Forest", "XGBoost", "Gradient Boosting"]:
                        explainer = shap.TreeExplainer(trained_model)
                        shap_values = explainer.shap_values(self.x_test)

                    elif model_name == "Linear Regression":
                        explainer = shap.LinearExplainer(trained_model, self.x_train)
                        shap_values = explainer.shap_values(self.x_test)

                    else:  # For SVR and other non-tree models
                        explainer = shap.KernelExplainer(trained_model.predict,
                                                         self.x_train[:50])  # Using a sample for efficiency
                        shap_values = explainer.shap_values(self.x_test[:50])  # Compute on a smaller test set

                    # Store trained model + predictions separately
                    model_storage[model_name] = {
                        "Trained Model": trained_model,
                        "Metrics": metrics,
                        "Predictions": predictions,  # Store predictions for this model
                        "Actual Values": self.y_test,  # Store actual values for this model
                        "Model Training": model_training,
                        "SHAP Values": shap_values,  # Store SHAP values
                    }
                    cv_scores = cross_validate(self.x_train, self.y_train)  # Cross-validation scores

                    # Store model summary
                    model_summary = {
                        "Dataset": title,
                        "Model": model_name,
                        "Best Params": best_params,
                        "MAE (min.)": metrics["MAE (min.)"],
                        "MAPE (%)": metrics["MAPE (%)"],
                        "RMSE (min.)": metrics["RMSE (min.)"],
                        "Mean CV R2": cv_scores.mean(),
                        "Std CV R2": cv_scores.std(),
                        "CV Scores": cv_scores.tolist(),
                    }
                    model_summaries.append(model_summary)

                # Save all models for this dataset
                self.final_models[title] = model_storage
                self.metrics_summary[title] = dataset_metrics

            except KeyError as e:
                print(f"Skipping DataFrame {i} due to missing column: {e}")

        return pd.DataFrame(model_summaries)

    def train_single_model(self, model_training, model_name):
        """
        Trains a single model, applying hyperparameter tuning if applicable.

        Returns:
        - trained_model: The trained ML model.
        - predictions: Model predictions on x_test.
        - best_params: Best hyperparameters if tuning was applied.
        """
        model_function = model_training.models[model_name]

        if self.param_grids and model_name in self.param_grids:
            param_grid = self.param_grids[model_name]
            model, _ = model_function()
            search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
            search.fit(self.x_train, self.y_train)
            trained_model = search.best_estimator_
            predictions = trained_model.predict(self.x_test)
            best_params = search.best_params_

        elif model_name == "Linear Regression":
            trained_model = LinearRegression()
            trained_model.fit(self.x_train, self.y_train)
            predictions = trained_model.predict(self.x_test)
            best_params = "Default"
            model_training.train_linear_model()

        else:
            trained_model, predictions = model_function()
            best_params = "Default"

        return trained_model, predictions, best_params

    def _setup_subplot_grid(self):
        """
        Creates a subplot grid based on the number of models and datasets.
        Returns the figure and axes.
        """
        num_plots = sum(len(models) for models in self.final_models.values())  # Count total models
        rows = (num_plots + 1) // 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
        axes = axes.flatten()
        plot_idx = 0
        return fig, axes, num_plots, plot_idx

    def plot_results(self):
        """
        Plots actual vs. predicted values for all models.
        """
        fig, axes, num_plots, plot_idx = self._setup_subplot_grid()

        for title, models in self.final_models.items():
            for model_name, model_info in models.items():
                predictions = model_info["Predictions"]

                ax = axes[plot_idx]
                ax.scatter(self.y_test, predictions, label="Predictions", s=20, alpha=0.6)
                ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()],
                        color="red", linestyle="-", label="Perfect Prediction")
                ax.set_title(f"{model_name} - {title}", fontsize=14)
                ax.set_xlabel("Actual Values (min.)")
                ax.set_ylabel("Predicted Values (min.)")
                ax.legend()
                plot_idx += 1

        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self):
        """
        Plots learning curves for all models and datasets.
        """
        fig, axes, num_plots, plot_idx = self._setup_subplot_grid()

        for title, models in self.final_models.items():
            for model_name, model_info in models.items():
                trained_model = model_info["Trained Model"]

                train_sizes, train_scores, test_scores = learning_curve(
                    trained_model, self.x_train, self.y_train, cv=5,
                    scoring="neg_mean_squared_error", train_sizes=np.linspace(0.1, 1.0, 10))
                train_mean = -np.mean(train_scores, axis=1)
                test_mean = -np.mean(test_scores, axis=1)

                ax = axes[plot_idx]
                ax.plot(train_sizes, train_mean, label="Training Score", color="blue")
                ax.plot(train_sizes, test_mean, label="Validation Score", color="orange")
                ax.set_title(f"Learning Curve: {model_name} - {title}", fontsize=14)
                ax.set_xlabel("Training Size")
                ax.set_ylabel("Error")
                ax.legend()
                plot_idx += 1

        plt.tight_layout()
        plt.show()

    def plot_shap_values(self):
        """
        Plots SHAP summary values for each trained model.
        """
        for title, models in self.final_models.items():
            for model_name, model_info in models.items():
                shap_values = model_info["SHAP Values"]
                x_test = model_info["Actual Values"]  # Ensure correct dataset

                if shap_values.shape[0] != x_test.shape[0]:
                    print(f"Skipping {model_name} - Mismatched SHAP Values: {shap_values.shape} vs {x_test.shape}")
                    continue  # Skip plotting if mismatch remains

                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, x_test, show=False)
                plt.title(f"SHAP Summary Plot: {model_name} - {title}")
                plt.show()
