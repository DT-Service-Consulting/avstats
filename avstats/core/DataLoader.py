import yaml
import json
import pandas as pd
from pathlib import Path
import os


class DataLoader:
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the DataLoader with a configuration file path.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config_path = Path(config_path)
        self.data_paths = None
        self._load_config()

    def _load_config(self):
        """Load the configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.data_paths = config['data_paths']
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except Exception as e:
            raise Exception(f"An error occurred while loading the config file: {e}")

    def load_data(self):
        """
        Load data from the specified paths in the configuration.

        Returns:
            tuple: A tuple containing DataFrames for avstats, passengers, and an airport mapping dictionary.
        """
        try:
            df_avstats = pd.read_csv(self.data_paths['avstats'])
            df_passengers = pd.read_excel(
                self.data_paths['passengers'],
                sheet_name='Sheet 1',
                header=8,
                engine='openpyxl'
            )
            with open(self.data_paths['airport_mapping'], 'r') as json_file:
                airport_mapping = json.load(json_file)
            return df_avstats, df_passengers, airport_mapping
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
        return None, None, None