import pytest
import pandas as pd
import numpy as np
from avstats.core.ML_workflow.TimeSeriesAnalysis import TimeSeriesAnalysis, adf_test
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


@pytest.fixture
def sample_data():
    data = {
        'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
        'value': np.random.randn(100).cumsum()
    }
    return pd.DataFrame(data)

@pytest.fixture
def tsa_instance(sample_data):
    return TimeSeriesAnalysis(
        df=sample_data,
        start_date='2020-01-01',
        end_date='2020-04-09',
        train_end=pd.Timestamp('2020-03-01'),
        test_end=pd.Timestamp('2020-03-31'),
        column='value',
        date_column='date'
    )

def test_adf_test_positive():
    series = np.random.randn(100).cumsum()
    p_value = adf_test(series)
    assert 0 <= p_value <= 1, "ADF test did not return a valid p-value"

def test_adf_test_invalid_input():
    with pytest.raises(ValueError):
        adf_test("not_a_series")

def test_check_stationarity(tsa_instance):
    differenced_data, is_stationary = tsa_instance.check_stationarity(max_diffs=2)
    assert isinstance(differenced_data, pd.Series), "Differenced data is not a pandas Series"
    assert isinstance(is_stationary, bool), "is_stationary is not a boolean"

def test_plot_acf_pacf(tsa_instance):
    tsa_instance.plot_acf_pacf(acf_lag=10, pacf_lag=10)
    # No assertions as this is a visual test

def test_plot_forecast(tsa_instance):
    predictions = pd.Series(
        np.random.randn(len(tsa_instance.df[tsa_instance.train_end + pd.Timedelta(days=1):tsa_instance.test_end])),
        index=tsa_instance.df[tsa_instance.train_end + pd.Timedelta(days=1):tsa_instance.test_end].index
    )
    tsa_instance.plot_forecast(predictions, title="Test Forecast Plot")
    # No assertions as this is a visual test

def test_arima_forecast(tsa_instance):
    order = (1, 1, 1)
    test_data, predictions, residuals, model = tsa_instance.arima_sarimax_forecast(order=order)
    assert isinstance(test_data, pd.Series), "Test data is not a pandas Series"
    assert isinstance(predictions, pd.Series), "Predictions are not a pandas Series"
    assert isinstance(residuals, pd.Series), "Residuals are not a pandas Series"
    assert hasattr(model, "summary"), "Model does not have a summary method, indicating it is not a fitted ARIMA model"

def test_sarimax_forecast(tsa_instance):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    test_data, predictions, residuals, model = tsa_instance.arima_sarimax_forecast(order=order, seasonal_order=seasonal_order)
    assert isinstance(test_data, pd.Series), "Test data is not a pandas Series"
    assert isinstance(predictions, pd.Series), "Predictions are not a pandas Series"
    assert isinstance(residuals, pd.Series), "Residuals are not a pandas Series"
    assert hasattr(model, "summary"), "Model does not have a summary method, indicating it is not a fitted SARIMAX model"

def test_rolling_forecast(tsa_instance):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    rolling_actual, rolling_predictions, residuals = tsa_instance.rolling_forecast(
        order=order, train_window=30, seasonal_order=seasonal_order
    )
    assert isinstance(rolling_actual, pd.Series), "Rolling actual values are not a pandas Series"
    assert isinstance(rolling_predictions, pd.Series), "Rolling predictions are not a pandas Series"
    assert isinstance(residuals, pd.Series), "Residuals are not a pandas Series"
