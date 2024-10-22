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