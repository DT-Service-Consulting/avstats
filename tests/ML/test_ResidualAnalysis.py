import pytest
import numpy as np
from avstats.core.ML_workflow.ResidualAnalysis import ResidualAnalysis
import matplotlib.pyplot as plt
from pydantic import ValidationError


@pytest.fixture
def sample_data():
    model = "SampleModel"
    y_pred = np.random.rand(100) * 20000  # Random predicted values in the range of 0-20000
    residuals = np.random.randn(100) * 1000  # Random residuals with normal distribution
    return model, y_pred, residuals


@pytest.fixture
def residual_analysis_instance(sample_data):
    model, y_pred, residuals = sample_data
    return ResidualAnalysis(model, y_pred, residuals)


def test_plot_residuals(residual_analysis_instance):
    model_name = "Test Dataset"

    # Test without subplot
    plt.figure()  # Ensure independent plot
    residual_analysis_instance.plot_residuals(dataset_name=model_name)
    plt.close()  # Close the figure to free memory

    # Test with subplot
    plt.figure()
    residual_analysis_instance.plot_residuals(dataset_name=model_name, subplot_position=(1, 2, 1))
    plt.close()


def test_q_q_normality(residual_analysis_instance):
    model_name = "Test Dataset"

    # Test without subplot
    plt.figure()  # Ensure independent plot
    residual_analysis_instance.q_q_normality(dataset_name=model_name)
    plt.close()  # Close the figure to free memory

    # Test with subplot
    plt.figure()
    residual_analysis_instance.q_q_normality(dataset_name=model_name, subplot_position=(1, 2, 1))
    plt.close()


def test_histogram_normality(residual_analysis_instance):
    model_name = "Test Dataset"

    # Test without subplot
    plt.figure()  # Ensure independent plot
    residual_analysis_instance.histogram_normality(dataset_name=model_name)
    plt.close()  # Close the figure to free memory

    # Test with subplot
    plt.figure()
    residual_analysis_instance.histogram_normality(dataset_name=model_name, subplot_position=(1, 2, 1))
    plt.close()


def test_attributes(sample_data):
    model, y_pred, residuals = sample_data
    analysis = ResidualAnalysis(model, y_pred, residuals)

    assert analysis.model == model, "Model attribute mismatch"
    assert np.array_equal(analysis.y_pred, y_pred), "Predicted values mismatch"
    assert np.array_equal(analysis.residuals, residuals), "Residuals mismatch"


def test_invalid_y_pred_type():
    model = "SampleModel"
    y_pred = [1, 2, 3, 4]  # Invalid type (list instead of np.ndarray)
    residuals = np.array([0.1, -0.2, 0.3, -0.4])

    with pytest.raises(ValueError, match="y_pred must be an instance of numpy.ndarray"):
        ResidualAnalysis(model, y_pred, residuals)

def test_invalid_residuals_type():
    model = "SampleModel"
    y_pred = np.array([1, 2, 3, 4])
    residuals = [0.1, -0.2, 0.3, -0.4]  # Invalid type (list instead of np.ndarray)

    with pytest.raises(ValueError, match="residuals must be an instance of numpy.ndarray"):
        ResidualAnalysis(model, y_pred, residuals)

def test_mismatched_lengths():
    model = "SampleModel"
    y_pred = np.array([1, 2, 3, 4])
    residuals = np.array([0.1, -0.2])  # Mismatched length

    with pytest.raises(ValueError, match="y_pred and residuals must have the same length"):
        ResidualAnalysis(model, y_pred, residuals)

def test_invalid_y_pred_values():
    model = "SampleModel"
    y_pred = np.array([-1, 25000, 15000, 30000])  # Out of range values
    residuals = np.array([0.1, -0.2, 0.3, -0.4])

    with pytest.raises(ValueError, match="y_pred values must be in the range \[0, 20000\]"):
        ResidualAnalysis(model, y_pred, residuals)
