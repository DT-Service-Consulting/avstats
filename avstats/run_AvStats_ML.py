import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime

# Ensure the logs directory exists
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

try:
    from core.config import PARAM_GRID_RF
    from core.ML.OneHotEncoding import OneHotEncoding
    from core.ML.DataPreparation import DataPreparation
    from core.ML.Multicollinearity import Multicollinearity
    from core.ML.ModelTraining import ModelTraining
    from core.ML.TimeSeriesAnalysis import TimeSeriesAnalysis
    from core.ML.NeuralNetworks import NeuralNetworks
    from core.ML.ModelEvaluation import cross_validate, evaluate_model, plot_combined
    from core.ML.ResidualAnalysis import ResidualAnalysis
except ModuleNotFoundError as e:
    logging.error(f"Module import error: {e}", exc_info=True)
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "modeling_pipeline.log")])

def main():
    try:
        logging.info("Starting Data Modeling and Predicting Pipeline...")

        # Load the dataset
        df_merged = pd.read_csv(os.path.join("..", "data", "df_merged.csv"))
        df_weather = pd.read_csv(os.path.join("..", "data", "df_weather.csv"))

        # Step 1: Encode routes
        data_encoding = OneHotEncoding(df_merged)
        df_encoded, _, _ = data_encoding.encode_routes()
        df_clean = data_encoding.clean_data()

        data_encoding_weather = OneHotEncoding(df_weather)
        df_encoded_weather, _, _ = data_encoding_weather.encode_routes()
        one_route_df = df_encoded_weather[pd.to_numeric(df_encoded_weather['BRU-MAD'], errors='coerce') == 2]
        logging.info("Routes encoded successfully.")

        # Step 2: Standardize & Regularize data
        data_prep = DataPreparation(df_clean, 'total_dep_delay')  # df_clean
        scaled_df, target_variable = data_prep.standardize_data()
        data_prep.select_important_features()
        logging.info("Data standardized and regularized.")

        # Step 3: Handle multicollinearity
        logging.info("Checking for multicollinearity...")
        df_vif_cleaned, _ = Multicollinearity(scaled_df, target_variable, verbose=False).remove_high_vif_features()

        # Step 4: Correlation Matrix
        correlation_matrix = df_vif_cleaned.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 7})
        plt.title('Correlation Matrix')
        plt.show()

        # Step 5: Split data & Initialize ModelTraining class
        x = df_vif_cleaned.drop(columns=['total_dep_delay'])  # Adjust target variable name if needed
        y = df_vif_cleaned['total_dep_delay']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets.")
        model_training = ModelTraining(x_train, y_train, x_test, y_test)
        evaluation_results = []

        # Step 6: Cross-validation
        cv = cross_validate(x_train, y_train)
        logging.info(f"Cross Validation: {cv}")

        # Step 7: Train and evaluate models
        for model_name, model_func in model_training.models.items():
            logging.info(f"{model_name} model fitting...")
            model, predictions = model_func()
            model_training.plot_model(model_name)
            residuals = y_test - predictions
            metrics = evaluate_model(y_test, predictions, residuals)
            evaluation_results.append({"Model": model_name, **metrics})

        """# Step 10: Hyperparameter tuning with Grid Search
        logging.info("Tuning hyperparameters with Grid Search...")
        grid_tuning_model, grid_tuning_parameters, grid_predictions, sample_sizes, grid_train_errors, grid_test_errors = model_training.tune_and_evaluate(
            param_grid=PARAM_GRID_RF,
            verbose=0,
            search_type='grid',
            log_scale=True  # Use log scale for sample sizes
        )
        print(f"Grid Search completed. Best Parameters: {grid_tuning_parameters}")
        grid_cv_mae, grid_cv_mape, grid_cv_rmse = evaluate_model(y_test, grid_predictions)"""

        # Step 11: ARIMA
        logging.info("ARIMA model fitting...")
        arima_analysis = TimeSeriesAnalysis(one_route_df, datetime(2023, 1, 1),
                                            datetime(2023, 12, 31),
                                            datetime(2023, 6, 30),
                                            datetime(2023, 12, 31), column='total_dep_delay')

        arima_test_data, arima_predictions, arima_residuals, arima_model = arima_analysis.arima_sarimax_forecast(order=(3, 1, 5))
        arima_metrics = evaluate_model(arima_test_data, arima_predictions, arima_residuals)
        evaluation_results.append({"Model": "ARIMA", **arima_metrics})

        # Step 12: SARIMAX
        logging.info("SARIMAX model fitting...")
        sarimax_test_data, sarimax_predictions, sarimax_residuals, sarimax_model = arima_analysis.arima_sarimax_forecast(order=(1, 0, 1), seasonal_order=(0, 1, 1, 30))
        sarimax_metrics = evaluate_model(sarimax_test_data, sarimax_predictions, sarimax_residuals)
        evaluation_results.append({"Model": "SARIMAX", **sarimax_metrics})

        # Step 13: Rolling Forecast Origin
        logging.info("Rolling Forecast Origin model fitting...")
        rolling_actual, rolling_predictions, rolling_residuals = arima_analysis.rolling_forecast(order=(1, 1, 1), train_window=180, seasonal_order=(0, 1, 1, 7))
        rolling_metrics = evaluate_model(rolling_actual, rolling_predictions, rolling_residuals)
        evaluation_results.append({"Model": "Rolling Forecast", **rolling_metrics})

        # Step 14: Model Comparison
        plot_combined("ARIMA", arima_test_data, arima_predictions, arima_residuals)
        plot_combined("SARIMAX", sarimax_test_data, sarimax_predictions, sarimax_residuals)
        plot_combined("Rolling Forecast", rolling_actual, rolling_predictions, rolling_residuals)
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_data = evaluation_df.set_index('Model')
        print(evaluation_df)

        # Visualizations
        evaluation_data[['MAE (min.)', 'RMSE (min.)']].plot(kind='bar', figsize=(12, 4), alpha=0.7)
        plt.title('Model Performance (MAE and RMSE)')
        plt.ylabel('(min.)')
        plt.xlabel('')
        plt.xticks(rotation=0, horizontalalignment='center')  # Rotate labels for better readability
        plt.legend(title="Metrics")
        plt.show()

        evaluation_data[['MAPE (%)']].plot(kind='bar', figsize=(12, 4), alpha=0.7)
        plt.title('Model Performance (MAPE)')
        plt.ylabel('(%)')
        plt.xlabel('')
        plt.ylim(0, 100)  # Set y-axis range from 0 to 100
        plt.xticks(rotation=0, horizontalalignment='center')  # Rotate labels for better readability
        plt.legend(title="Metrics")
        plt.show()

        """# Step 14:
        logging.info("Neural Network model fitting...")
        nn = NeuralNetworks(df_weather)
        nn_actual, nn_predictions = nn.neural_networks()
        nn_residuals = nn_actual - nn_predictions
        nn_metrics = evaluate_model(nn_actual, nn_predictions, nn_residuals)
        evaluation_results.append({"Model": "Neural Networks", **nn_metrics})
        """
    except Exception as e:
        logging.error(f"Error in Modeling and Predicting Pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
