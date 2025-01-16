import pytest
import numpy as np
import pandas as pd
from avstats.core.ML.ModelTraining import ModelTraining


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    np.random.seed(42)
    x_train = pd.DataFrame(np.random.rand(100, 3), columns=['Feature1', 'Feature2', 'Feature3']).to_numpy()
    y_train = pd.Series(np.random.rand(100)).to_numpy()
    x_test = pd.DataFrame(np.random.rand(20, 3), columns=['Feature1', 'Feature2', 'Feature3']).to_numpy()
    y_test = pd.Series(np.random.rand(20)).to_numpy()
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
    """Test the model's plotting functionality with metrics."""
    x_train, y_train, x_test, y_test = sample_data
    model_trainer = ModelTraining(x_train, y_train, x_test, y_test)

    # Train a random forest model to generate predictions
    rf_model, y_pred = model_trainer.train_random_forest()

    # Mock evaluation metrics
    evaluation_metrics = {
        "MAE": 0.123,
        "RMSE": 0.234,
        "R^2": 0.789
    }

    # Assign predictions and actual values to the instance
    model_trainer.y_test = y_test
    model_trainer.y_pred = y_pred

    # Ensure no exceptions are raised during plotting
    try:
        model_trainer.plot_model(
            title="Random Forest: Predicted vs Actual Values",
            evaluation_metrics=evaluation_metrics
        )
    except Exception as e:
        pytest.fail(f"Plotting failed with exception: {e}")
