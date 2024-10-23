# avstats
### README

# Data Cleaning and Multicollinearity Analysis

This project provides tools for data cleaning and multicollinearity analysis using Python. It includes functionalities to check for missing values, identify duplicate rows, and calculate Variance Inflation Factor (VIF) to assess multicollinearity.

## Features

- **Data Cleaning**: Identify missing values and duplicate rows.
- **Multicollinearity Analysis**: Calculate VIF and remove features with high VIF values.

## Requirements

- Python 3.8+
- Poetry

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/your-repo.git
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

## Usage

### Data Cleaning

The `DataCleaning` class provides methods to check for missing values and duplicate rows.

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
import pandas as pd
from avstats.classes import Multicollinearity

data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 4, 6, 8, 10],
    'target': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
multicollinearity = Multicollinearity(dataframe=df)

vif_data = multicollinearity.calculate_vif()
print(f"VIF data:\n{vif_data}")

cleaned_df = multicollinearity.remove_high_vif_features(target_variable='target')
print(f"Cleaned DataFrame:\n{cleaned_df}")
```

## Running Tests

Tests are written using `pytest`. To run the tests, use the following command:

```sh
pytest
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.