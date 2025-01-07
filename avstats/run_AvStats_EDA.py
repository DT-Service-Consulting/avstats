import logging
import os
import sys
from pathlib import Path
import warnings

# Ensure the logs directory exists
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

try:
    from core.DataLoader import DataLoader
    from core.EDA.DataProcessing import DataProcessing
    from core.EDA.FlightPerformance import FlightPerformance
    from core.EDA.WeatherData import WeatherData
    from core.EDA.MergeData import MergeData
    from core.EDA.PassengerData import PassengerData
    from avstats.core.EDA_utils import *
    from avstats.core.general_utils import *
except ModuleNotFoundError as e:
    logging.error(f"Module import error: {e}", exc_info=True)
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / 'weather_pipeline.log')])

def main():
    try:
        logging.info("Starting Exploratory Data Analysis Pipeline...")

        # Load the dataset
        data_loader = DataLoader(config_path='config.yaml')
        df_avstats, df_passengers, airport_mapping = data_loader.load_data()

        if df_avstats is None or df_passengers is None or airport_mapping is None:
            raise SystemExit("Data files are missing or failed to load. Please check your configuration.")

        # Step 1: Data Processing
        data_processing = DataProcessing(df_avstats, unique_column='uuid')
        df = data_processing.preprocess_avstats()
        logging.info(f"Data was processed successfully")

        # Step 2: Check for missing values
        quality_metrics = data_processing.check_missing_and_duplicates(df)
        logging.info(f"Missing Values: {quality_metrics['missing_values']}, "
                     f"Duplicate Rows: {quality_metrics['duplicate_rows']}, "
                     f"Missing by Column: {quality_metrics['missing_by_column']}")

        # Step 3: Performance Summary
        flight_performance = FlightPerformance(df)
        performance_metrics = flight_performance.overall_performance()
        logging.info("Overall Flight Performance Percentage: ")
        for metric, value in performance_metrics.items():
            logging.info(f"{metric}: {value:.2f}%")

        # Step 4: Delay Ranges
        delay_ranges = [(0, 60), (60, 120), (120, 180), (180, float('inf'))]
        delay_summary = flight_performance.delay_ranges_summary(delay_ranges)
        logging.info("\nDelay Summary:")
        for range_label, percentage in delay_summary.items():
            logging.info(f"Delays {range_label}: {percentage:.2f}%")

        # Step 5: Meteostat (weather)
        logging.info("Fetching weather data from the Meteostat library...")
        warnings.filterwarnings("ignore", category=FutureWarning)
        weather_fetcher = WeatherData(df)
        weather_fetcher.assign_coordinates().head()
        weather_fetcher.fetch_weather_data()
        df_weather_merged = weather_fetcher.merge_weather_with_flights()
        logging.info("Weather data fetched successfully.")

        # Step 6: Cleaning and merging weather data
        df_weather_merged = df_weather_merged.drop_duplicates()
        df_weather_merged = df_weather_merged.dropna(subset=['tavg_dep', 'tavg_arr'])
        df_weather_merged = df_weather_merged.apply(
            lambda col: col.astype(str) if col.dtype.name == 'category' else col)
        df_weather_merged.fillna(0, inplace=True)
        aggregator = MergeData(df_weather_merged)
        aggregator.preprocess_datetime()
        df_grouped_daily = aggregator.aggregate_daily(passenger_data=False)
        logging.info("Weather data merged successfully.")
        save_dataframe(df_grouped_daily, "df_weather")

        # Step 7: Passenger data
        passengers = PassengerData(df_passengers, airport_mapping)
        df_passengers_cleaned = passengers.process_passenger_data()
        aggregator.aggregate_daily(passenger_data=True)
        df_merged = aggregator.aggregate_passengers(df_passengers_cleaned)
        logging.info("Passenger data fetched and merged successfully.")
        save_dataframe(df_merged, "df_merged")

    except Exception as e:
        logging.error(f"Error in Flight and Weather Data Processing Pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
