from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent.parent

# Paths to various resources
DATA_DIR = BASE_DIR / 'data'

# Path to the Excel file
EXCEL_FILE = DATA_DIR / 'avia_par_be_page_spreadsheet.xlsx'

# Path to the CSV file
CSV_FILE_SCHED = DATA_DIR / 'avStats_schedule_historical_2023_BRU.csv'
CSV_FILE_WEATHER = DATA_DIR / 'BRU2023_weather.csv'

# Hyperparameter grid for Random Forest model
PARAM_GRID_RF = {
    'n_estimators': [100, 200, 300],       # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],      # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],      # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],        # Minimum number of samples required to be at a leaf node
}