import logging
import pandas as pd
import matplotlib.pyplot as plt
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
    from core.ML_workflow.ModelComparison import ModelComparison
    from core.ML_workflow.ResidualAnalysis import ResidualAnalysis
    from core.ML_workflow.ModelEvaluation import cross_validate, evaluate_model
    from core.ML_workflow.TimeSeriesAnalysis import TimeSeriesAnalysis
    from core.ML_workflow.NeuralNetworks import NeuralNetworks
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

def preprocess_data(df, target_column):
    """
    Preprocess data: One-hot encoding, cleaning, standardization, and multicollinearity handling.
    """
    logging.info("Preprocessing data...")
    data_encoding = OneHotEncoding(df)
    df_encoded, _, _ = data_encoding.encode_routes()
    df_clean = data_encoding.clean_data()

    # Standardization
    data_prep = DataPreparation(df_clean, target_column)
    scaled_df, target_variable = data_prep.standardize_data()

    # Handle multicollinearity
    multicollinearity = Multicollinearity(scaled_df, target_variable)
    df_vif_cleaned, _ = multicollinearity.remove_high_vif_features()

    return df_vif_cleaned, target_variable

def main():
    try:
        logging.info("Starting Data Modeling and Predicting Pipeline...")

        # Load datasets
        df_merged = pd.read_csv(DATA_DIR / "df_merged.csv")
        df_weather = pd.read_csv(DATA_DIR / "df_weather.csv")

        # Validate input
        required_columns = ['total_dep_delay', 'Date', 'BRU-MAD']
        for col in required_columns:
            if col not in df_weather.columns:
                raise ValueError(f"Missing required column: {col}")

        # Preprocess data
        df_vif_cleaned, target_variable = preprocess_data(df_merged, 'total_dep_delay')

        # Split data
        x = df_vif_cleaned.drop(columns=['total_dep_delay'])
        y = df_vif_cleaned['total_dep_delay']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Initialize model training
        model_training = ModelTraining(x_train, y_train, x_test, y_test)

        # Train and evaluate models
        evaluation_results = []

        # Linear Regression
        linear_model, linear_predictions = model_training.train_linear_model()
        linear_residuals = y_test - linear_predictions
        linear_mae, linear_mape, linear_rmse = evaluate_model(y_test, linear_predictions, linear_residuals)
        evaluation_results.append({"Model": "Linear Regression", "MAE (min.)": linear_mae, "MAPE (%)": linear_mape, "RMSE (min.)": linear_rmse})

        # Decision Tree
        decision_model, decision_predictions = model_training.train_decision_tree()
        decision_residuals = y_test - decision_predictions
        decision_mae, decision_mape, decision_rmse = evaluate_model(y_test, decision_predictions, decision_residuals)
        evaluation_results.append({"Model": "Decision Tree", "MAE (min.)": decision_mae, "MAPE (%)": decision_mape, "RMSE (min.)": decision_rmse})

        # Random Forest with Hyperparameter Tuning
        rf_model, rf_predictions = model_training.train_random_forest(param_grid=PARAM_GRID_RF, perform_tuning=True)
        rf_residuals = y_test - rf_predictions
        rf_mae, rf_mape, rf_rmse = evaluate_model(y_test, rf_predictions, rf_residuals)
        evaluation_results.append({"Model": "Random Forest", "MAE (min.)": rf_mae, "MAPE (%)": rf_mape, "RMSE (min.)": rf_rmse})

        # Time-Series Models
        #stepwise_fit = auto_arima(df_arima['total_dep_delay'], seasonal=True, m=7)
        #print(stepwise_fit.summary())
        arima_analysis = TimeSeriesAnalysis(df_weather, datetime(2023, 1, 1), datetime(2023, 12, 31), datetime(2023, 6, 30), datetime(2023, 12, 31), column='total_dep_delay', date_column='Date')
        arima_test_data, arima_predictions, arima_residuals, _ = arima_analysis.arima_sarimax_forecast(order=(3, 1, 5))
        arima_mae, arima_mape, arima_rmse = evaluate_model(arima_test_data, arima_predictions, arima_residuals)
        evaluation_results.append({"Model": "ARIMA", "MAE (min.)": arima_mae, "MAPE (%)": arima_mape, "RMSE (min.)": arima_rmse})

        # Neural Networks
        nn = NeuralNetworks(df_weather)
        nn_actual, nn_predictions = nn.neural_networks()
        nn_residuals = nn_actual - nn_predictions
        nn_mae, nn_mape, nn_rmse = evaluate_model(nn_actual, nn_predictions, nn_residuals)
        evaluation_results.append({"Model": "Neural Networks", "MAE (min.)": nn_mae, "MAPE (%)": nn_mape, "RMSE (min.)": nn_rmse})

        # Model Comparison
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.to_csv(DATA_DIR / "evaluation_results.csv", index=False)

        # Visualization
        evaluation_df[['MAE (min.)', 'RMSE (min.)']].plot(kind='bar', figsize=(12, 6))
        plt.title('Model Performance (MAE and RMSE)')
        plt.show()

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
