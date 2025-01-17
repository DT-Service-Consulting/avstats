import pandas as pd
from avstats.core.ML.NeuralNetworks import NeuralNetworks
import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from avstats.core.ML.NeuralNetworks import nn_plots


@pytest.fixture
def mock_dataframe():
    """Fixture for providing mock DataFrame."""
    np.random.seed(42)
    data = {'total_dep_delay': np.random.randint(1, 100, 100)}
    return pd.DataFrame(data)


def test_create_dataset():
    """Test the create_dataset method."""
    data = np.array([[i] for i in range(10)])
    lookback = 3

    x, y = NeuralNetworks.create_dataset(data, lookback)

    # Check the shape of x and y
    assert x.shape == (6, 3), "x shape is incorrect"
    assert y.shape == (6,), "y shape is incorrect"

    # Verify content
    expected_x = np.array([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7]
    ])
    expected_y = np.array([3, 4, 5, 6, 7, 8])

    assert np.array_equal(x, expected_x), "x content is incorrect"
    assert np.array_equal(y, expected_y), "y content is incorrect"


@patch("avstats.core.ML.NeuralNetworks.Sequential")
@patch("avstats.core.ML.NeuralNetworks.MinMaxScaler")
def test_neural_networks(mock_scaler, mock_sequential, mock_dataframe):
    """Test the neural_networks method of NeuralNetworks class."""
    # Mock the MinMaxScaler behavior
    scaler_instance = mock_scaler.return_value
    scaler_instance.fit_transform.side_effect = lambda x: x / np.max(x)
    scaler_instance.inverse_transform.side_effect = lambda x: x * np.max(mock_dataframe['total_dep_delay'])

    # Mock the Sequential model
    model_instance = mock_sequential.return_value
    model_instance.predict.side_effect = lambda x: np.zeros((x.shape[0], 1))
    model_instance.fit.return_value = None

    # Mock evaluation metrics to match the actual keys and structure
    def mock_evaluate_model(_, __, ___):
        return {"MAE (min.)": 57.28, "MAPE (%)": 100.0, "RMSE (min.)": 62.77}

    # Patch evaluate_model function
    with patch("avstats.core.ML.ModelEvaluation.evaluate_model", mock_evaluate_model):
        # Instantiate NeuralNetworks and run the neural_networks method
        nn = NeuralNetworks(mock_dataframe, column="total_dep_delay", look_back=10)
        model, y_test, predictions, metrics = nn.neural_networks()

        # Assertions
        assert model_instance is model, "Returned model instance is incorrect."
        assert isinstance(y_test, np.ndarray), "y_test should be a numpy array."
        assert isinstance(predictions, np.ndarray), "predictions should be a numpy array."
        assert isinstance(metrics, dict), "metrics should be a dictionary."
        assert y_test.shape == predictions.shape, "Shapes of y_test and predictions should match."

        # Verify metrics content
        expected_metrics = {"MAE (min.)": 57.28, "MAPE (%)": 100.0, "RMSE (min.)": 62.77}
        assert metrics == expected_metrics, "Metrics content is incorrect."


@patch("avstats.core.ML.NeuralNetworks.metrics_box")
def test_nn_plots(mock_metrics_box):
    """Test the nn_plots function."""
    np.random.seed(42)
    actual = np.random.rand(20)
    predicted = np.random.rand(20)
    metrics = {"MAE": 0.123, "RMSE": 0.234, "R^2": 0.789}

    # Create a single Axes for testing
    fig, ax = plt.subplots(figsize=(8, 6))

    # Call nn_plots directly
    nn_plots(ax, 0, actual, predicted, "LSTM Model", metrics)

    # Check if metrics_box was called
    mock_metrics_box.assert_called_once_with(metrics, ax)

    # Close the plot after testing
    plt.close(fig)

