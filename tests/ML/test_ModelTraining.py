import pytest
import numpy as np
import pandas as pd
from avstats.core.ML_workflow.ModelTraining import ModelTraining


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    np.random.seed(42)
    x_train = pd.DataFrame(np.random.rand(100, 3), columns=['Feature1', 'Feature2', 'Feature3'])
    y_train = pd.Series(np.random.rand(100))
    x_test = pd.DataFrame(np.random.rand(20, 3), columns=['Feature1', 'Feature2', 'Feature3'])
    y_test = pd.Series(np.random.rand(20))
    return x_train, y_train, x_test, y_test


def test_model_initialization(sample_data):
    """Test if ModelTraining initializes correctly with valid data."""
    x_train, y_train, x_test, y_test = sample_data
    try:
        model_trainer = ModelTraining(x_train, y_train, x_test, y_test)
        assert model_trainer.x_train.shape == x_train.shape, "x_train shape mismatch after initialization"
        assert model_trainer.y_train.shape == y_train.shape, "y_train shape mismatch after initialization"
        assert model_trainer.x_test.shape == x_test.shape, "x_test shape mismatch after initialization"
        assert model_trainer.y_test.shape == y_test.shape, "y_test shape mismatch after initialization"
    except Exception as e:
        pytest.fail(f"Initialization failed with exception: {e}")


def test_train_linear_model(sample_data):
    """Test training a linear model."""
    x_train, y_train, x_test, y_test = sample_data
    model_trainer = ModelTraining(x_train, y_train, x_test, y_test)

    ols_model, y_pred = model_trainer.train_linear_model()

    assert ols_model is not None, "OLS model should not be None"
    assert isinstance(y_pred, np.ndarray), "Predictions should be a numpy array"
    assert len(y_pred) == len(y_test), "Predictions length should match test set length"


def test_train_decision_tree(sample_data):
    """Test training a Decision Tree model."""
    x_train, y_train, x_test, y_test = sample_data
    model_trainer = ModelTraining(x_train, y_train, x_test, y_test)

    tree_model, y_pred = model_trainer.train_decision_tree()

    assert tree_model is not None, "Decision Tree model should not be None"
    assert isinstance(y_pred, np.ndarray), "Predictions should be a numpy array"
    assert len(y_pred) == len(y_test), "Predictions length should match test set length"


def test_train_random_forest(sample_data):
    """Test training a Random Forest model."""
    x_train, y_train, x_test, y_test = sample_data
    model_trainer = ModelTraining(x_train, y_train, x_test, y_test)

    rf_model, y_pred = model_trainer.train_random_forest()

    assert rf_model is not None, "Random Forest model should not be None"
    assert isinstance(y_pred, np.ndarray), "Predictions should be a numpy array"
    assert len(y_pred) == len(y_test), "Predictions length should match test set length"


def test_plot_model(sample_data):
    """Test the model's plotting functionality."""
    x_train, y_train, x_test, y_test = sample_data
    model_trainer = ModelTraining(x_train, y_train, x_test, y_test)

    # Train a random forest model to generate predictions
    model_trainer.train_random_forest()

    # Ensure no exceptions are raised during plotting
    try:
        model_trainer.plot_model("Random Forest: Predicted vs Actual Values")
    except Exception as e:
        pytest.fail(f"Plotting failed with exception: {e}")


def test_tune_and_evaluate(sample_data):
    """Test hyperparameter tuning and evaluation."""
    x_train, y_train, x_test, y_test = sample_data
    model_trainer = ModelTraining(x_train, y_train, x_test, y_test)

    param_grid = {
        'max_depth': [2, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    best_model, best_params, y_pred, sample_sizes, train_errors, test_errors = model_trainer.tune_and_evaluate(
        param_grid=param_grid,
        verbose=0,
        search_type='grid',
        cv=3
    )

    assert best_model is not None, "Best model should not be None"
    assert isinstance(best_params, dict), "Best parameters should be a dictionary"
    assert isinstance(y_pred, np.ndarray), "Predictions should be a numpy array"
    assert len(y_pred) == len(y_test), "Predictions length should match test set length"
    assert len(sample_sizes) == len(train_errors) == len(
        test_errors), "Sample sizes, train errors, and test errors lengths should match"

    # Ensure errors are non-negative
    assert all(e >= 0 for e in train_errors), "All training errors should be non-negative"
    assert all(e >= 0 for e in test_errors), "All test errors should be non-negative"
