import pytest
import numpy as np
from avstats.core.ML_workflow.ResidualAnalysis import ResidualAnalysis
import matplotlib.pyplot as plt


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
