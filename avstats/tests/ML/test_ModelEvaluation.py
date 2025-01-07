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


def test_cross_validate(sample_data):
    """
    Test the cross-validate function.
    """
    x_train, y_train = sample_data
    x_train_np = x_train.values
    y_train_np = y_train.values

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

    # Convert to numpy arrays for compatibility with the updated function
    x_np = x.values
    y_np = y.values

    # Simulate predictions using a simple OLS model
    ols_model = sm.OLS(y_np, sm.add_constant(x_np)).fit()
    predictions = ols_model.predict(sm.add_constant(x_np))
    residuals = y_np - predictions

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


def test_plot_combined(sample_data):
    """
    Test the plot_combined function.
    """
    x, y = sample_data

    # Convert to numpy arrays for compatibility with the updated function
    x_np = x.values
    y_np = y.values

    # Simulate predictions using a simple OLS model
    ols_model = sm.OLS(y_np, sm.add_constant(x_np)).fit()
    predictions = ols_model.predict(sm.add_constant(x_np))
    residuals = y_np - predictions

    # Ensure the function executes without error
    try:
        plt.figure()  # Ensure plot does not interfere with other tests
        plot_combined("OLS Model", actual=y_np, predicted=predictions, residuals=residuals)
        plt.close()
    except Exception as e:
        pytest.fail(f"plot_combined function raised an exception: {e}")
