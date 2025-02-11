# DataLoader.py
import yaml
import json
import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict


def save_dataframe(df, filename):
    # Define the path to save the file
    data_folder = os.path.join("..", "data")  # Adjust the path if needed
    file_path = os.path.join(data_folder, f"{filename}.csv")

    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")

def save_json(data, filename):
    # Define the path to save the file
    data_folder = os.path.join("..", "data")  # Adjust the path if needed
    file_path = os.path.join(data_folder, f"{filename}.json")

    # Save the JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"JSON file saved to {file_path}")

class DataLoader:
    def __init__(self, config_path: str = 'config.yaml') -> None:
        """
        Initialize the DataLoader with a configuration file path.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config_path = Path(config_path)
        self.data_paths = None
        self._load_config()

    def _load_config(self) -> None:
        """
        Load the configuration file to extract data paths.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            Exception: For other issues while loading the configuration.
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.data_paths = config['data_paths']
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except Exception as e:
            raise Exception(f"An error occurred while loading the config file: {e}")

    def load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, str]]]:
        """
        Load data from the specified paths in the configuration.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: DataFrame for aviation statistics (avstats).
                - pd.DataFrame: DataFrame for passenger data (passengers).
                - Dict[str, str]: Dictionary for airport mappings.
            If an error occurs, returns (None, None, None).
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
