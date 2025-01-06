import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from avstats.core.ML_workflow.ModelEvaluation import cross_validate, evaluate_model


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = pd.DataFrame(np.random.rand(100, 3), columns=['feature1', 'feature2', 'feature3'])
    y = x['feature1'] * 3 + x['feature2'] * 2 - x['feature3'] + np.random.normal(0, 0.1, 100)
    return x, y

def test_cross_validate(sample_data):
    x_train, y_train = sample_data

    # Convert to numpy arrays for compatibility with updated function
    x_train_np = x_train.values
    y_train_np = y_train.values

    cv_scores = cross_validate(x_train_np, y_train_np, cv=5)

    # Check that the function returns an array of length equal to the number of folds
    assert len(cv_scores) == 5

    # Verify the R2 scores are within a reasonable range (-1 to 1 for R2 score)
    assert all(-1 <= score <= 1 for score in cv_scores)

    # Test reproducibility with the same random seed
    cv_scores_repeated = cross_validate(x_train_np, y_train_np, cv=5)
    np.testing.assert_array_almost_equal(cv_scores, cv_scores_repeated)

def test_evaluate_model(sample_data):
    x, y = sample_data

    # Convert to numpy arrays for compatibility with updated function
    x_np = x.values
    y_np = y.values

    # Simulate predictions using a simple OLS model
    ols_model = sm.OLS(y_np, sm.add_constant(x_np)).fit()
    predictions = ols_model.predict(sm.add_constant(x_np))
    residuals = y_np - predictions

    mae, mape, rmse = evaluate_model(y_np, predictions, residuals)

    # Validate the computed metrics
    expected_mae = mean_absolute_error(y_np, predictions)
    expected_rmse = root_mean_squared_error(y_np, predictions)
    expected_mape = np.mean(abs(residuals / y_np)) * 100

    assert np.isclose(mae, expected_mae, rtol=1e-5)
    assert np.isclose(rmse, expected_rmse, rtol=1e-5)
    assert np.isclose(mape, expected_mape, rtol=1e-5)

    # Test behavior when residuals are not provided
    mae_no_residuals, mape_no_residuals, rmse_no_residuals = evaluate_model(y_np, predictions)

    assert np.isclose(mae_no_residuals, expected_mae, rtol=1e-5)
    assert np.isclose(rmse_no_residuals, expected_rmse, rtol=1e-5)
    assert mape_no_residuals is None

