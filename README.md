# avstats
### README

# Flight Delay Forecasting Package

This repository provides a comprehensive pipeline for forecasting flight delays, utilizing data cleaning, exploratory data analysis (EDA), and machine learning (ML) models. The package is designed to work with historical flight, weather, and passenger data to analyze patterns, build predictive models, and refine performance through hyperparameter tuning.

## Table of Contents

- [Project](#project)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Notebooks](#notebooks)
- - [Notebook 1 - Data Cleaning & EDA](#EDANotebook)
  - [Notebook 2 - Modeling and Predictions](#MLNotebook)
- [Usage](#usage)
- [Running Tests](#tests)
- [License](#license)


## Project

This project aims to develop a forecasting model for commercial flight delays. The objective is to analyze historical flight data and investigate how delays are influenced by factors such as flight routes, weather conditions, and passenger volumes. The expected outcome is to generate daily or weekly predictions of flight delays.

## Features

- **Data Cleaning & Processing**:
    - Identifies and removes duplicates and handles missing values.
    - Creates new features like latitude and longitude for airport locations.
    - Merges external data, including weather and passenger datasets, with flight schedules for comprehensive analysis.

- **Exploratory Data Analysis (EDA)**:
    - Analyzes flight status, performance, and time windows for trends.
    - Compares flight categories and evaluates delay statistics by airline.
    - Identifies and visualizes frequent routes and airports.

- **Modeling and Prediction**:
    - Supports machine learning models: Linear Regression, Decision Tree, and Random Forest.
    - Performs data preprocessing techniques like standardization, normalization, and regularization.
    - Mitigates multicollinearity using Variance Inflation Factor (VIF) analysis.

- **Hyperparameter Tuning and Model Evaluation**:
    - Uses Grid Search and Randomized Search for hyperparameter tuning.
    - Conducts residual analysis to assess model performance and validity.

## Requirements

- python = 3.12
- pandas = 2.2.3
- numpy = 1.26.4
- scipy = 1.14.1
- matplotlib = 3.9.2
- seaborn = 0.13.2
- jupyter = 1.1.1
- statsmodels = 0.14.4
- scikit-learn = 1.5.2
- pytest = 8.3.3
- meteostat = 1.6.8
- tensorflow = 2.18.0
- pydantic = 2.0.0


## Installation

To use this package:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your_username/flight-delay-forecasting.git
    cd your-repo
    ```

2. **Install Poetry**:
    Follow the instructions on the [Poetry website](https://python-poetry.org/docs/#installation) to install Poetry.


3. **Install dependencies**:
    ```sh
    poetry install
    ```

4. **Activate the virtual environment**:
    ```sh
    poetry shell
    ```

## Notebooks

The project includes two main Jupyter notebooks that guide you through data preparation, EDA, and model development.

### EDANotebook

*Notebook 1 - Data Cleaning & EDA*

This notebook handles the initial stages of data processing and analysis.

- **Data Cleaning & Processing**:
    - Detects duplicates and missing values, creates new features, and cleans data from multiple sources.
    - Adds weather data (from the Meteostat library) based on route coordinates and date.
    - Integrates passenger data and merges all relevant information.

- **Exploratory Data Analysis (EDA)**:
    - Analyzes flight status, performance, time windows, flight categories, and frequent routes/airports.
    - Explores delay patterns by airline and examines the impact of external factors on delays.


### MLNotebook

*Notebook 2 - Modeling and Predictions*

**Note**: Run the EDA notebook first to generate necessary cleaned data files.

This notebook includes feature engineering, ML model training, and performance evaluation.

- **Feature Engineering**:
    - Applies one-hot encoding and correlation analysis to refine features.
    - Uses standardization, normalization, and regularization techniques to improve model performance.

- **Modeling**:
    - Implements multiple models: Linear Regression, Decision Tree, and Random Forest.
    - Includes hyperparameter tuning with Grid Search CV and Randomized Search CV for optimal model selection.

- **Residual Analysis**:
    - Assesses residuals to ensure model assumptions are met, covering key diagnostics such as linearity, homoscedasticity, normality, and independence.

## Usage

To run the analysis pipeline:

1. Start with Notebook 1 (Data Cleaning & EDA) to clean and prepare the data.
2. Proceed to Notebook 2 (Modeling and Predictions) for model training and evaluation.
3. Use the provided classes in ML_pipelines/ for a more customized or automated workflow.

To tune hyperparameters and get the best model:

```python
import pandas as pd
from avstats.core.EDA import DataProcessing

data = {
    'id': [1, 2, 2, 4, 5],
    'value': [10, 20, 20, None, 50]
}
df = pd.DataFrame(data)
cleaner = DataProcessing(unique_column='id')

missing_values, duplicate_rows = cleaner.check_missing_and_duplicates(df)
print(f"Missing values: {missing_values}")
print(f"Duplicate rows:\n{duplicate_rows}")
```

## Tests

Tests are written using `pytest`. To run the tests, use the following command:

```sh
pytest
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
