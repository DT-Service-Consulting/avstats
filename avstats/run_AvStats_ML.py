import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.tree import export_text, plot_tree

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
    from core.ML_workflow.TimeSeriesAnalysis import TimeSeriesAnalysis
    from core.ML_workflow.NeuralNetworks import NeuralNetworks
    from core.ML_workflow.ModelEvaluation import cross_validate, evaluate_model, plot_combined
    from core.ML_workflow.ResidualAnalysis import ResidualAnalysis
except ModuleNotFoundError as e:
    logging.error(f"Module import error: {e}", exc_info=True)
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
        df_merged = pd.read_csv(os.path.join("..", "data", "df_merged.csv"))
        df_weather = pd.read_csv(os.path.join("..", "data", "df_weather.csv"))
        logging.info("Dataset loaded successfully.")

        # Step 1: Encode routes
        data_encoding = OneHotEncoding(df_merged)
        df_encoded, corr_df, route_columns = data_encoding.encode_routes()
        df_clean = data_encoding.clean_data()

        df_encoded_weather, corr_df_weather, route_columns_weather = OneHotEncoding(df_weather).encode_routes()
        df_encoded_weather['BRU-MAD'] = pd.to_numeric(df_encoded_weather['BRU-MAD'], errors='coerce')
        one_route_df = df_encoded_weather[df_encoded_weather['BRU-MAD'] == 2]
        logging.info("Routes encoded successfully.")

        # Step 3: Standardize data
        data_prep = DataPreparation(df_clean, 'total_dep_delay')  # df_clean
        scaled_df, target_variable = data_prep.standardize_data()
        logging.info("Data standardized.")

        # Step 4: Regularize data
        important_features_df, important_features = data_prep.select_important_features()
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        important_features.plot(kind='bar')
        plt.title('Top Important Features Based on Lasso Coefficients')
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.show()

        # Step 5: Handle multicollinearity
        logging.info("Checking for multicollinearity...")
        multicollinearity = Multicollinearity(scaled_df, target_variable, verbose=False)
        df_vif_cleaned, _ = multicollinearity.remove_high_vif_features()
        correlation_matrix = df_vif_cleaned.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 7})
        plt.title('Correlation Matrix')
        plt.show()

        # Step 6: Split data & Initialize ModelTraining class
        x = df_vif_cleaned.drop(columns=['total_dep_delay'])  # Adjust target variable name if needed
        y = df_vif_cleaned['total_dep_delay']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets.")
        model_training = ModelTraining(x_train, y_train, x_test, y_test)
        evaluation_results = []

        # Step 7: Cross-validation
        cv = cross_validate(x_train, y_train)
        logging.info(f"Cross Validation: {cv}")

        # Step 8: Linear Regression
        logging.info("Linear Regression model fitting...")
        linear_model, linear_predictions = model_training.train_linear_model()
        model_training.plot_model("Linear Regression")
        linear_residuals = y_test - linear_predictions
        linear_mae, linear_mape, linear_rmse = evaluate_model(y_test, linear_predictions, linear_residuals)
        evaluation_results.append({"Model": "Linear Regression",
                                   **{"MAE (min.)": linear_mae, "MAPE (%)": linear_mape, "RMSE (min.)": linear_rmse}})

        # Step 9: Decision Tree
        logging.info("Decision Tree model fitting...")
        decision_model, decision_predictions = model_training.train_decision_tree()
        model_training.plot_model("Decision Tree")
        decision_residuals = y_test - decision_predictions
        decision_mae, decision_mape, decision_rmse = evaluate_model(y_test, decision_predictions, decision_residuals)
        evaluation_results.append({"Model": "Decision Tree", **{"MAE (min.)": decision_mae, "MAPE (%)": decision_mape,
                                                                "RMSE (min.)": decision_rmse}})
        plt.figure(figsize=(12, 8))
        plot_tree(decision_model, feature_names=x.columns, filled=True, rounded=True)
        plt.show()
        export_text(decision_model, feature_names=list(x.columns))

        # Step 10: Random Forest
        logging.info("Random Forest model fitting...")
        random_f_model, random_f_predictions = model_training.train_random_forest()
        model_training.plot_model("Random Forest")
        random_f_residuals = y_test - random_f_predictions
        random_f_mae, random_f_mape, random_f_rmse = evaluate_model(y_test, random_f_predictions, random_f_residuals)
        evaluation_results.append({"Model": "Random Forest", **{"MAE (min.)": random_f_mae, "MAPE (%)": random_f_mape, "RMSE (min.)": random_f_rmse}})
        """
        # Step 10: Hyperparameter tuning with Grid Search
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

        # Step 11: ARIMA
        logging.info("ARIMA model fitting...")
        arima_analysis = TimeSeriesAnalysis(one_route_df, datetime(2023, 1, 1), datetime(2023, 12, 31), datetime(2023, 6, 30), datetime(2023, 12, 31), column='total_dep_delay', date_column='Date')
        arima_test_data, arima_predictions, arima_residuals, arima_model = arima_analysis.arima_sarimax_forecast(order=(3, 1, 5))
        arima_mae, arima_mape, arima_rmse = evaluate_model(arima_test_data, arima_predictions, arima_residuals)
        evaluation_results.append({"Model": "ARIMA", "MAE (min.)": arima_mae, "MAPE (%)": arima_mape, "RMSE (min.)": arima_rmse})

        # Step 12: SARIMAX
        logging.info("SARIMAX model fitting...")
        sarimax_test_data, sarimax_predictions, sarimax_residuals, sarimax_model = arima_analysis.arima_sarimax_forecast(
            order=(1, 0, 1), seasonal_order=(0, 1, 1, 30))
        sarimax_mae, sarimax_mape, sarimax_rmse = evaluate_model(sarimax_test_data, sarimax_predictions, sarimax_residuals)
        evaluation_results.append({"Model": "SARIMAX", "MAE (min.)": sarimax_mae, "MAPE (%)": sarimax_mape, "RMSE (min.)": sarimax_rmse})

        # Step 13: Rolling Forecast Origin
        logging.info("Rolling Forecast Origin model fitting...")
        rolling_actual, rolling_predictions, rolling_residuals = arima_analysis.rolling_forecast(
            order=(1, 1, 1), train_window=180, seasonal_order=(0, 1, 1, 7)
        )
        rolling_mae, rolling_mape, rolling_rmse = evaluate_model(rolling_actual, rolling_predictions, rolling_residuals)
        evaluation_results.append({"Model": "Rolling Forecast", "MAE (min.)": rolling_mae, "MAPE (%)": rolling_mape, "RMSE (min.)": rolling_rmse})

        # Step 14: Model Comparison
        plot_combined("ARIMA", arima_test_data, arima_predictions, arima_residuals)
        plot_combined("SARIMAX", sarimax_test_data, sarimax_predictions, sarimax_residuals)
        plot_combined("Rolling Forecast", rolling_actual, rolling_predictions, rolling_residuals)
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_data = evaluation_df.set_index('Model')

        # Visualizations
        evaluation_data[['MAE (min.)', 'RMSE (min.)']].plot(kind='bar', figsize=(12, 4), alpha=0.7)
        plt.title('Model Performance (MAE and RMSE)')
        plt.show()
        evaluation_data[['MAPE (%)']].plot(kind='bar', figsize=(12, 4), alpha=0.7)
        plt.title('Model Performance (MAPE)')
        plt.show()

        """
        # Step 14:
        logging.info("Neural Network model fitting...")
        nn = NeuralNetworks(df_weather)
        nn_actual, nn_predictions = nn.neural_networks()
        nn_residuals = nn_actual - nn_predictions
        nn_mae, nn_mape, nn_rmse = evaluate_model(nn_actual, nn_predictions, nn_residuals)
        evaluation_results.append({"Model": "Neural Networks", **{"MAE (min.)": nn_mae, "MAPE (%)":nn_mape, "RMSE (min.)": nn_rmse}})
        """
    except Exception as e:
        logging.error(f"Error in Modeling and Predicting Pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
