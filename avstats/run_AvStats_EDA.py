import logging
import pandas as pd
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add the base path to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / 'logs'
DATA_DIR = BASE_DIR / 'data'
LOG_DIR.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist
sys.path.append(str(BASE_DIR / 'core'))

try:
    from core.config import PARAM_GRID_RF
    from core.EDA_workflow.CheckMissingValues import DataCleaning
    from core.EDA_workflow.WeatherData import WeatherData
    from core.EDA_workflow.FlightPerformance import FlightPerformance
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
        logging.FileHandler(LOG_DIR / 'flight_weather_pipeline.log')
    ]
)

def main():
    try:
        logging.info("Starting Flight and Weather Data Processing Pipeline...")

        # Load the dataset
        file_path = DATA_DIR / "df_merged.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")

        # Step 1: Data Cleaning
        cleaner = DataCleaning(unique_column='uuid')
        missing_values, duplicate_rows, missing_by_column = cleaner.check_missing_and_duplicates(df)
        logging.info(f"Missing Values: {missing_values}, Duplicate Rows: {len(duplicate_rows)}")

        # Step 2: Assign Coordinates
        weather_data = WeatherData(df)
        df_with_coords = weather_data.assign_coordinates()
        logging.info("Coordinates assigned successfully.")

        # Step 3: Fetch and Merge Weather Data
        weather_data.fetch_weather_data()
        df_weather_merged = weather_data.merge_weather_with_flights()
        logging.info("Weather data merged successfully.")

        # Step 4: Analyze Flight Performance
        flight_perf = FlightPerformance(df_weather_merged)
        performance_metrics = flight_perf.overall_performance()
        logging.info(f"Overall Flight Performance: {performance_metrics}")

        logging.info("Flight and Weather Data Processing Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Error in Flight and Weather Data Processing Pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
