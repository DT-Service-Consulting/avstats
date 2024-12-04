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
    from core.ML_workflow.OneHotEncoding import DataPreparation
    from core.ML_workflow.Multicollinearity import Multicollinearity
    from core.ML_workflow.ModelTraining import ModelTraining
    from core.ML_workflow.ResidualAnalysis import ResidualAnalysis
    from core.ML_workflow.ModelEvaluation import ModelEvaluation
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
        file_path = DATA_DIR / "df_merged.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        df_merged = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")

        # Initialize DataPreparation
        data_prep = DataPreparation(df_merged)

        # Step 1: Encode routes
        df_encoded, _, _ = data_prep.encode_routes()
        logging.info("Routes encoded successfully.")

        # Step 2: Clean data by removing columns with only zeros
        df_clean = data_prep.clean_data()
        logging.info("Data cleaned successfully.")

        # Step 3: Standardize data
        scaled_df, target_variable = data_prep.standardize_data()
        logging.info("Data standardized.")

        # Step 4: Regularize data
        important_features_df, _ = data_prep.select_important_features()

        # Step 5: Handle multicollinearity
        logging.info("Checking for multicollinearity...")
        multicollinearity = Multicollinearity(scaled_df, target_variable, verbose=False)
        df_vif_cleaned, _ = multicollinearity.remove_high_vif_features()

        # Step 6: Split data
        x = df_vif_cleaned.drop(columns=['total_dep_delay'])  # Adjust target variable name if needed
        y = df_vif_cleaned['total_dep_delay']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets.")

        # Step 7: Initialize ModelTraining class
        model_training = ModelTraining(x_train, y_train, x_test, y_test)

        # Step 8: Hyperparameter tuning with Grid Search
        logging.info("Tuning hyperparameters with Grid Search...")
        grid_tuning_model, grid_tuning_parameters, grid_predictions = model_training.tune_and_evaluate(
            param_grid=PARAM_GRID_RF,
            verbose=0,
            search_type='grid'
        )
        logging.info(f"Grid Search completed. Best Parameters: {grid_tuning_parameters}")
        model_training.plot_model()

        # Step 9: Evaluate model
        grid_evaluation = ModelEvaluation(grid_tuning_model, grid_predictions, x_train, y_train, x_test, y_test)
        grid_metrics = grid_evaluation.evaluate_model()
        logging.info(f"Model Evaluation Metrics: {grid_metrics}")

        # Step 10: Cross validation
        grid_cv = grid_evaluation.cross_validate()
        logging.info(f"5-fold Cross Validation: {grid_cv}")

        # Step 11: Perform residual analysis
        residual_analysis = ResidualAnalysis(grid_tuning_model, grid_predictions, x_test, y_test)
        residual_analysis.plot_residuals()
        logging.info("Residual analysis completed.")

        logging.info("Data Modeling and Predicting Pipeline completed successfully! "
                     "Log information can be found within the logs/modeling_pipeline.log file.")

    except Exception as e:
        logging.error(f"Error in Modeling and Predicting Pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

