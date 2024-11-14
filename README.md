# avstats
### README

# Flight Delay Forecasting Package

This repository provides a comprehensive pipeline for forecasting flight delays, utilizing data cleaning, exploratory data analysis (EDA), and machine learning (ML) models. The package is designed to work with historical flight, weather, and passenger data to analyze patterns, build predictive models, and refine performance through hyperparameter tuning.

## Table of Contents

- [Project Overview](#projectoverview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Notebooks](#notebooks)
- - [Notebook 1 - Data Cleaning & EDA](#notebook1)
  - [Notebook 2 - Modeling and Predictions](#notebook2)
- [Usage](#usage)
- [Running Tests](#test)
- [License](#license)


## Project Overview

This project is part of an analysis and predictive modeling series focused on U.S. small airlines, leveraging various data sources. It aims to:

    Clean and preprocess flight data, integrating additional weather and passenger information.
    Explore and visualize flight data to uncover insights into delay patterns and key features.
    Develop and optimize machine learning models to forecast delays with high accuracy.
    

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
- scipy = 1.14.1
- matplotlib = 3.9.2
- seaborn = 0.13.2
- jupyter = 1.1.1
- statsmodels = 0.14.4
- scikit-learn = 1.5.2
- pytest = 8.3.3
- meteostat = 1.6.8

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

### Notebook 1 - Data Cleaning & EDA

This notebook handles the initial stages of data processing and analysis.

1. **Data Cleaning & Processing**:
- Detects duplicates and missing values, creates new features, and cleans data from multiple sources.
- Adds weather data (from the Meteostat library) based on route coordinates and date.
- Integrates passenger data and merges all relevant information.

2. **Exploratory Data Analysis (EDA)**:
- Analyzes flight status, performance, time windows, flight categories, and frequent routes/airports.
- Explores delay patterns by airline and examines the impact of external factors on delays.


### Notebook 2 - Modeling and Predictions

**Note**: Run the EDA notebook first to generate necessary cleaned data files.

This notebook includes feature engineering, ML model training, and performance evaluation.

1. **Feature Engineering**:
- Applies one-hot encoding and correlation analysis to refine features.
- Uses standardization, normalization, and regularization techniques to improve model performance.

2. **Modeling**:
- Implements multiple models: Linear Regression, Decision Tree, and Random Forest.
        Includes hyperparameter tuning with Grid Search CV and Randomized Search CV for optimal model selection.

3. **Residual Analysis**:
- Assesses residuals to ensure model assumptions are met, covering key diagnostics such as linearity, homoscedasticity, normality, and independence.

## Usage

To run the analysis pipeline:

1. Start with Notebook 1 (Data Cleaning & EDA) to clean and prepare the data.
2. Proceed to Notebook 2 (Modeling and Predictions) for model training and evaluation.
3. Use the provided classes in ML_pipelines/ for a more customized or automated workflow.

To tune hyperparameters and get the best model:

```python
import pandas as pd
from avstats.classes import DataCleaning

data = {
    'id': [1, 2, 2, 4, 5],
    'value': [10, 20, 20, None, 50]
}
df = pd.DataFrame(data)
cleaner = DataCleaning(unique_column='id')

missing_values, duplicate_rows = cleaner.check_missing_and_duplicates(df)
print(f"Missing values: {missing_values}")
print(f"Duplicate rows:\n{duplicate_rows}")
```

### Multicollinearity Analysis

The `Multicollinearity` class provides methods to calculate VIF and remove features with high VIF values.

```python
from ML_pipelines.ModelTraining import ModelTraining

# Define parameter grid for tuning
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize and tune the model
model_training = ModelTraining(x_train, y_train, x_test, y_test)
best_model, best_params = model_training.tune_and_evaluate(
    param_grid=param_grid_rf,
    search_type='grid'
)
print(f"Best Model: {best_model}")
print(f"Best Parameters: {best_params}")
```

## Running Tests

Tests are written using `pytest`. To run the tests, use the following command:

```sh
pytest
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
