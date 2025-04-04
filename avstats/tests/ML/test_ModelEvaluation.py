import pytest
from avstats.core.ML.ModelEvaluation import *


@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    """
    np.random.seed(42)
    x = pd.DataFrame(np.random.rand(100, 3), columns=['feature1', 'feature2', 'feature3'])
    y = x['feature1'] * 3 + x['feature2'] * 2 - x['feature3'] + np.random.normal(0, 0.1, 100)
    return x, y


def prepare_ols_predictions(x, y):
    """
    Helper function to prepare predictions and residuals using an OLS model.
    """
    x_np = x.values
    y_np = y.values
    ols_model = sm.OLS(y_np, sm.add_constant(x_np)).fit()
    predictions = ols_model.predict(sm.add_constant(x_np))
    residuals = y_np - predictions
    return x_np, y_np, predictions, residuals


def test_cross_validate(sample_data):
    """
    Test the cross-validate function.
    """
    x_train, y_train = sample_data
    x_train_np, y_train_np, _, _ = prepare_ols_predictions(x_train, y_train)

    cv_scores = cross_validate(x_train_np, y_train_np, cv=5)
    assert len(cv_scores) == 5

    # Verify the R2 scores are within a reasonable range (-1 to 1 for R2 score)
    assert all(-1 <= score <= 1 for score in cv_scores)

    # Test reproducibility with the same random seed
    cv_scores_repeated = cross_validate(x_train_np, y_train_np, cv=5)
    np.testing.assert_array_almost_equal(cv_scores, cv_scores_repeated)


def test_evaluate_model(sample_data):
    """
    Test the evaluate_model function.
    """
    x, y = sample_data
    _, y_np, predictions, residuals = prepare_ols_predictions(x, y)

    # Evaluate model with residuals
    metrics = evaluate_model(y_np, predictions, residuals)

    # Validate the computed metrics
    expected_mae = round(mean_absolute_error(y_np, predictions), 2)
    expected_rmse = round(root_mean_squared_error(y_np, predictions), 2)
    expected_mape = round(np.mean(abs(residuals / y_np)) * 100, 2)

    assert metrics["MAE (min.)"] == expected_mae, f"Expected MAE: {expected_mae}, Got: {metrics['MAE (min.)']}"
    assert metrics["RMSE (min.)"] == expected_rmse, f"Expected RMSE: {expected_rmse}, Got: {metrics['RMSE (min.)']}"
    if metrics["MAPE (%)"] is not None:
        assert metrics["MAPE (%)"] == expected_mape, f"Expected MAPE: {expected_mape}, Got: {metrics['MAPE (%)']}"

    # Test behavior when residuals are not provided
    metrics_no_residuals = evaluate_model(y_np, predictions)

    assert metrics_no_residuals["MAE (min.)"] == expected_mae
    assert metrics_no_residuals["RMSE (min.)"] == expected_rmse
    assert metrics_no_residuals["MAPE (%)"] is None


def test_metrics_box():
    """
    Test the metrics_box function.
    """
    evaluation_metrics = {
        "MAE (min.)": 2.5,
        "RMSE (min.)": 3.0,
        "MAPE (%)": 10.5
    }

    try:
        plt.figure()  # Create a new figure
        metrics_box(evaluation_metrics)  # Ensure the function runs without errors
        plt.close()  # Close the plot after the test
    except Exception as e:
        pytest.fail(f"metrics_box function raised an exception: {e}")


def test_plot_combined(sample_data):
    """
    Test the plot_combined function.
    """
    x, y = sample_data
    _, y_np, predictions, residuals = prepare_ols_predictions(x, y)

    # Ensure the function executes without error
    try:
        plt.figure()  # Ensure plot does not interfere with other tests
        plot_combined("OLS Model", actual=y_np, predicted=predictions, residuals=residuals)
        plt.close()
    except Exception as e:
        pytest.fail(f"plot_combined function raised an exception: {e}")


def test_plot_metrics():
    """
    Test the plot_metrics function to ensure it can plot model performance metrics correctly.
    """
    # Mock evaluation results
    evaluation_results = [
        {"Model": "Model A", "MAE (min.)": 2.5, "RMSE (min.)": 3.0, "MAPE (%)": 10.5},
        {"Model": "Model B", "MAE (min.)": 3.1, "RMSE (min.)": 4.2, "MAPE (%)": 12.8},
        {"Model": "Model C", "MAE (min.)": 2.8, "RMSE (min.)": 3.5, "MAPE (%)": None},  # Handle None values for MAPE
    ]

    # Ensure no exceptions are raised during plotting
    try:
        plt.figure()
        plot_metrics(evaluation_results)
        plt.close()  # Close the figure to free memory
    except Exception as e:
        pytest.fail(f"Plotting failed with exception: {e}")
