import logging
import pandas as pd
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime

# Add the base path to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / 'logs'
DATA_DIR = BASE_DIR / 'data'
LOG_DIR.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist
sys.path.append(str(BASE_DIR / 'core'))

try:
    from core.config import PARAM_GRID_RF
    from core.ML_workflow.OneHotEncoding import OneHotEncoding
    from core.ML_workflow.DataPreparation import DataPreparation
    from core.ML_workflow.Multicollinearity import Multicollinearity
    from core.ML_workflow.ModelTraining import ModelTraining
    from core.ML_workflow.ResidualAnalysis import ResidualAnalysis
    from core.ML_workflow.ModelEvaluation import cross_validate, evaluate_model
    from core.ML_workflow.TimeSeriesAnalysis import TimeSeriesAnalysis, NeuralNetworks
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print("Check if the path to the 'core' directory and the modules inside it are correct.")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / 'modeling_pipeline.log')
    ]
)

def main():
    try:
        logging.info("Starting Data Modeling and Predicting Pipeline...")

        # Load the dataset
        file_path = os.path.join("..", "data", "df_merged.csv")
        file_path_weather = os.path.join("..", "data", "df_weather.csv")
        df_merged = pd.read_csv(file_path)
        df_weather = pd.read_csv(file_path_weather)
        logging.info("Dataset loaded successfully.")

        # Step 1: Encode routes
        data_encoding = OneHotEncoding(df_merged)
        df_encoded, corr_df, route_columns = data_encoding.encode_routes()
        df_clean = data_encoding.clean_data()

        data_encoding_weather = OneHotEncoding(df_weather)
        df_encoded_weather, corr_df_weather, route_columns_weather = data_encoding_weather.encode_routes()
        df_encoded_weather['BRU-MAD'] = pd.to_numeric(df_encoded_weather['BRU-MAD'], errors='coerce')
        filtered_df = df_encoded_weather[df_encoded_weather['BRU-MAD'] == 2]
        logging.info("Routes encoded successfully.")

        # Step 3: Standardize data
        data_prep = DataPreparation(df_clean, 'total_dep_delay')  # df_clean
        scaled_df, target_variable = data_prep.standardize_data()
        logging.info("Data standardized.")

        # Step 4: Regularize data
        important_features_df, _ = data_prep.select_important_features()

        # Step 5: Handle multicollinearity
        logging.info("Checking for multicollinearity...")
        multicollinearity = Multicollinearity(scaled_df, target_variable, verbose=False)
        df_vif_cleaned, _ = multicollinearity.remove_high_vif_features()

        # Step 6: Split data & Initialize ModelTraining class
        x = df_vif_cleaned.drop(columns=['total_dep_delay'])  # Adjust target variable name if needed
        y = df_vif_cleaned['total_dep_delay']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets.")
        model_training = ModelTraining(x_train, y_train, x_test, y_test)

        # Step 7: Linear Regression
        logging.info("Linear Regression model fitting...")
        linear_model, linear_predictions = model_training.train_linear_model()
        model_training.plot_model()
        linear_residuals = y_test - linear_predictions
        linear_mae, linear_mape, linear_rmse = evaluate_model(y_test, linear_predictions, linear_residuals)
        linear_cv = cross_validate(x_train, y_train)
        #logging.info(f"MAE: {linear_mae}, MPAE: {linear_mape}, RMSE: {linear_rmse}")
        logging.info(f"Cross Validation: {linear_cv}")

        # Step 8: Random Forest
        logging.info("Random Forest model fitting...")
        random_forest_model, random_forest_predictions = model_training.train_random_forest()
        model_training.plot_model()
        random_forest_residuals = y_test - random_forest_predictions
        random_forest_mae, random_forest_mape, random_forest_rmse = evaluate_model(y_test, random_forest_predictions,
                                                                                   random_forest_residuals)
        random_forest_cv = cross_validate(x_train, y_train)
        #logging.info(f"MAE: {random_forest_mae}, MPAE: {random_forest_mape}, RMSE: {random_forest_rmse}")
        logging.info(f"Cross Validation: {random_forest_cv}")

        # Step 9: Hyperparameter tuning with Grid Search
        logging.info("Tuning hyperparameters with Grid Search...")
        grid_tuning_model, grid_tuning_parameters, grid_predictions, sample_sizes, grid_train_errors, grid_test_errors = model_training.tune_and_evaluate(
            param_grid=PARAM_GRID_RF,
            verbose=0,
            search_type='grid',
            log_scale=True  # Use log scale for sample sizes
        )
        print(f"Grid Search completed. Best Parameters: {grid_tuning_parameters}")
        grid_cv_mae, grid_cv_mape, grid_cv_rmse = evaluate_model(y_test, grid_predictions)

        """
        # Step 10: ARIMA
        logging.info("ARIMA model fitting...")
        df_arima = filtered_df.copy()
        df_arima['Date'] = pd.to_datetime(df_arima['Date'])
        df_arima.set_index('Date', inplace=True)
        df_arima = df_arima.resample('D').ffill()
        df_arima.reset_index(inplace=True)

        train_end = datetime(2023, 6, 30)
        test_end = datetime(2023, 12, 31)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        arima_analysis = TimeSeriesAnalysis(df_arima, start_date, end_date, train_end, test_end,
                                            column='total_dep_delay', date_column='Date')
        arima_test_data, arima_predictions, arima_residuals, arima_model = arima_analysis.arima_sarimax_forecast(
            order=(3, 1, 5))  # 3, 1, 3 or 2, 1, 3
        arima_mae, arima_mape, arima_rmse = evaluate_model(arima_test_data, arima_predictions, arima_residuals)

        # Step 11: SARIMAX
        logging.info("SARIMAX model fitting...")
        sarimax_test_data, sarimax_predictions, sarimax_residuals, sarimax_model = arima_analysis.arima_sarimax_forecast(
            order=(1, 0, 1), seasonal_order=(0, 1, 1, 30))
        sarimax_mae, sarimax_mape, sarimax_rmse = evaluate_model(sarimax_test_data, sarimax_predictions,
                                                                 sarimax_residuals)

        # Step 12: Rolling Forecast Origin
        logging.info("Rolling Forecast Origin model fitting...")
        rolling_actual, rolling_predictions, rolling_residuals = arima_analysis.rolling_forecast(
            order=(1, 1, 1), train_window=180, seasonal_order=(0, 1, 1, 7)
        )
        rolling_mae, rolling_mape, rolling_rmse = evaluate_model(rolling_actual, rolling_predictions, rolling_residuals)

        # Step 13:
        logging.info("Neural Network model fitting...")
        nn = NeuralNetworks(df_weather)
        nn_actual, nn_predictions = nn.neural_networks()
        nn_residuals = nn_actual - nn_predictions
        nn_mae, nn_mape, nn_rmse = evaluate_model(nn_actual, nn_predictions, nn_residuals)
        """
    except Exception as e:
        logging.error(f"Error in Modeling and Predicting Pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

