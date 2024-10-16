from imports import *

class DataLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path if file_path else r'C:\4.DTSC_Programs\4.TransStatsData\AvStats\avStats_schedule_historical_2023_BRU.csv'
        self.dataframe = self.load_data()

    def load_data(self):
        """Load the data from a CSV file."""
        try:
            df = pd.read_csv(self.file_path)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None