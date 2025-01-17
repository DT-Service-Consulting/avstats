import pytest
import numpy as np
import pandas as pd
from avstats.core.ML.TimeSeriesAnalysis import TimeSeriesAnalysis, adf_test


@pytest.fixture
def sample_data():
    # Create a DataFrame with a 'Date' column and random time series data
    date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = {
        'Date': date_range,
        'value': np.random.randn(100).cumsum()
    }
    return pd.DataFrame(data)


@pytest.fixture
def tsa_instance(sample_data):
    return TimeSeriesAnalysis(
        df=sample_data,
        start_date=pd.Timestamp('2020-01-01'),
        end_date=pd.Timestamp('2020-04-09'),
        train_end=pd.Timestamp('2020-03-01'),
        test_end=pd.Timestamp('2020-03-31'),
        column='value'
    )


def test_adf_test_positive():
    series = np.random.randn(100).cumsum()
    p_value = adf_test(series)
    assert 0 <= p_value <= 1, "ADF test did not return a valid p-value"


def test_adf_test_invalid_input():
    with pytest.raises(ValueError, match="Input series must be a pandas Series, list, or numpy array."):
        adf_test("not_a_series")


def test_time_series_analysis_validation(sample_data):
    valid_instance = TimeSeriesAnalysis(
        df=sample_data,
        start_date=pd.Timestamp('2020-01-01'),
        end_date=pd.Timestamp('2020-04-09'),
        train_end=pd.Timestamp('2020-03-01'),
        test_end=pd.Timestamp('2020-03-31'),
        column='value'
    )
    assert isinstance(valid_instance, TimeSeriesAnalysis), "Failed to validate a valid instance"

    with pytest.raises(ValueError, match="start_date must be earlier than end_date"):
        TimeSeriesAnalysis(
            df=sample_data,
            start_date=pd.Timestamp('2020-04-01'),
            end_date=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-03-01'),
            test_end=pd.Timestamp('2020-03-31'),
            column='value'
        )


def test_check_stationarity(tsa_instance):
    differenced_data, is_stationary = tsa_instance.check_stationarity(max_diffs=2)
    assert isinstance(differenced_data, pd.Series), "Differenced data is not a pandas Series"
    assert isinstance(is_stationary, bool), "is_stationary is not a boolean"
    assert len(differenced_data) > 0, "Differenced data is empty"


def test_plot_acf_pacf(tsa_instance):
    try:
        tsa_instance.plot_acf_pacf(acf_lag=10, pacf_lag=10)
    except Exception as e:
        pytest.fail(f"Plotting ACF/PACF failed with exception: {e}")


def test_plot_forecast(tsa_instance):
    predictions = pd.Series(
        np.random.randn(len(tsa_instance.df[tsa_instance.train_end + pd.Timedelta(days=1):tsa_instance.test_end])),
        index=tsa_instance.df[tsa_instance.train_end + pd.Timedelta(days=1):tsa_instance.test_end].index
    )
    try:
        tsa_instance.plot_forecast(predictions, title="Test Forecast Plot", metrics={"MAE": 0.1, "RMSE": 0.2})
    except Exception as e:
        pytest.fail(f"Plotting forecast failed with exception: {e}")


def test_arima_forecast(tsa_instance):
    order = (1, 1, 1)
    model, test_data, predictions, residuals, metrics = tsa_instance.arima_sarimax_forecast(order=order)
    assert isinstance(test_data, pd.Series), "Test data is not a pandas Series"
    assert isinstance(predictions, pd.Series), "Predictions are not a pandas Series"
    assert isinstance(residuals, pd.Series), "Residuals are not a pandas Series"
    assert "MAE" in metrics, "Metrics do not contain MAE"


def test_sarimax_forecast(tsa_instance):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    model, test_data, predictions, residuals, metrics = tsa_instance.arima_sarimax_forecast(order=order, seasonal_order=seasonal_order)
    assert isinstance(test_data, pd.Series), "Test data is not a pandas Series"
    assert isinstance(predictions, pd.Series), "Predictions are not a pandas Series"
    assert isinstance(residuals, pd.Series), "Residuals are not a pandas Series"
    assert "MAE" in metrics, "Metrics do not contain MAE"


def test_rolling_forecast(tsa_instance):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    train_window = 30
    rolling_actual, rolling_predictions, residuals, metrics = tsa_instance.rolling_forecast(
        order=order, train_window=train_window, seasonal_order=seasonal_order
    )
    assert isinstance(rolling_actual, np.ndarray), "Rolling actual values should be a numpy array"
    assert isinstance(rolling_predictions, np.ndarray), "Rolling predictions should be a numpy array"
    assert isinstance(residuals, np.ndarray), "Residuals should be a numpy array"
    assert len(rolling_actual) > 0, "Rolling forecast did not produce any actual values"
    assert len(rolling_predictions) > 0, "Rolling forecast did not produce any predictions"
