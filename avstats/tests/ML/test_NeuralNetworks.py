import pytest
import pandas as pd
import numpy as np
from avstats.core.ML.NeuralNetworks import NeuralNetworks
from unittest.mock import patch


@pytest.fixture
def mock_dataframe():
    """Fixture for providing mock DataFrame."""
    data = {
        'total_dep_delay': np.random.randint(1, 100, 100)
    }
    return pd.DataFrame(data)

def test_create_dataset():
    # Test the create_dataset method
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

    # Instantiate NeuralNetworks and run the neural_networks method
    nn = NeuralNetworks(mock_dataframe)
    y_test, predictions = nn.neural_networks()

    # Assertions
    assert y_test.shape == predictions.shape, "Shapes of y_test and predictions should match."
