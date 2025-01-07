import pytest
from avstats.core.ML.TimeSeriesAnalysis import *


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
    with pytest.raises(ValueError):
        adf_test("not_a_series")


def test_time_series_analysis_validation(sample_data):
    # Test validation through Pydantic
    valid_instance = TimeSeriesAnalysis(
        df=sample_data,
        start_date=pd.Timestamp('2020-01-01'),
        end_date=pd.Timestamp('2020-04-09'),
        train_end=pd.Timestamp('2020-03-01'),
        test_end=pd.Timestamp('2020-03-31'),
        column='value'
    )
    assert isinstance(valid_instance, TimeSeriesAnalysis), "Failed to validate a valid instance"

    # Invalid date range
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
    """
    Test the rolling_forecast function.
    """
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    train_window = 30

    # Execute the rolling forecast
    rolling_actual, rolling_predictions, residuals = tsa_instance.rolling_forecast(
        order=order, train_window=train_window, seasonal_order=seasonal_order
    )

    # Validate the outputs
    assert isinstance(rolling_actual, np.ndarray), "Rolling actual values should be a numpy array"
    assert isinstance(rolling_predictions, np.ndarray), "Rolling predictions should be a numpy array"
    assert isinstance(residuals, np.ndarray), "Residuals should be a numpy array"

    # Ensure lengths match
    assert len(rolling_actual) == len(rolling_predictions), "Actual and predicted lengths do not match"
    assert len(residuals) == len(rolling_predictions), "Residuals length does not match predictions length"

    # Check residuals computation
    computed_residuals = rolling_actual - rolling_predictions
    np.testing.assert_array_almost_equal(
        residuals, computed_residuals, decimal=5, err_msg="Residuals are not correctly computed"
    )

    # Ensure rolling_forecast produces at least one prediction
    assert len(rolling_predictions) > 0, "Rolling forecast did not produce any predictions"

    # Test the visual component (plot)
    try:
        plt.figure()
        # Adjust the index range to match the predictions length
        plot_index = tsa_instance.df[tsa_instance.column].index[train_window:train_window + len(rolling_predictions)]
        tsa_instance.plot_forecast(
            pd.Series(rolling_predictions, index=plot_index),
            title="Rolling Forecast vs Actual",
        )
        plt.close()
    except Exception as e:
        pytest.fail(f"Plotting failed with exception: {e}")
