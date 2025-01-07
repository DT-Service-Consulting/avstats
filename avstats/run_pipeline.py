import logging
import pandas as pd
import os
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from avstats.core.ML.ModelTraining import ModelTraining
from avstats.core.ML.ModelEvaluation import cross_validate, evaluate_model
from avstats.core.ML.OneHotEncoding import OneHotEncoding
from avstats.core.ML.DataPreparation import DataPreparation
from avstats.core.ML.Multicollinearity import Multicollinearity
from avstats.core.visualization.visualization_utils import plot_metrics

# Load configurations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Ensure the logs directory exists
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "pipeline.log")])

def process(merged_file, weather_file, config):
    df_merged = pd.read_csv(merged_file)
    df_weather = pd.read_csv(weather_file)
    config = config

    # Encode routes with passenger
    data_encoding = OneHotEncoding(df_merged)
    df_encoded, _, _ = data_encoding.encode_routes()
    df_clean = data_encoding.clean_data()

    # Encode routes without passengers
    data_encoding_weather = OneHotEncoding(df_weather)
    df_encoded_weather, _, _ = data_encoding_weather.encode_routes()

    # Filter specific route
    focus_route = config["routes"]["focus_route"]
    one_route_df = df_encoded_weather[pd.to_numeric(df_encoded_weather[focus_route], errors="coerce") == 2]

    # Standardization and multicollinearity handling
    data_prep = DataPreparation(df_clean, "total_dep_delay")
    scaled_df, target_variable = data_prep.standardize_data()
    df_vif_cleaned, _ = Multicollinearity(scaled_df, target_variable).remove_high_vif_features()

    # Train-test split
    x = df_vif_cleaned.drop(columns=["total_dep_delay"])
    y = df_vif_cleaned["total_dep_delay"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config["hyperparameters"]["test_size"],
                                                        random_state=config["hyperparameters"]["random_state"])
    return x_train, x_test, y_train, y_test, one_route_df


def main():
    try:
        logging.info("Starting pipeline...")

        # Step 1: Data Preprocessing
        x_train, x_test, y_train, y_test, one_route_df = process(
            os.path.join("..", "data", "df_merged.csv"),os.path.join("..", "data", "df_weather.csv"), config)

        # Step 2: Model Training
        model_training = ModelTraining(x_train, y_train, x_test, y_test)
        evaluation_results = []

        # Step 3: Cross-validation
        cv = cross_validate(x_train, y_train)
        logging.info(f"Cross Validation: {cv}")

        # Train and evaluate models
        for model_name, model_func in model_training.models.items():
            model, predictions = model_func()
            model_training.plot_model(model_name)
            residuals = y_test - predictions
            metrics, mae, mape, rmse = evaluate_model(y_test, predictions, residuals)
            evaluation_results.append({"Model": model_name, "MAE (min.)": mae, "MAPE (%)": mape, "RMSE (min.)": rmse})

        # Step 3: Visualizations
        plot_metrics(evaluation_results)
        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
