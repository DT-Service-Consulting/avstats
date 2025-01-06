# core/ML_workflow/TimeSeriesAnalysis.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
from typing import Tuple, Optional, List, Union
from avstats.core.ML_workflow.validators_ML.validator_time_series_analysis import TimeSeriesAnalysisInput


def adf_test(series: Union[pd.Series, List[float], np.ndarray]) -> float:
    """
    Perform Augmented Dickey-Fuller test and return the p-value.

    Args:
        series (Union[pd.Series, List[float], np.ndarray]): The time series to test for stationarity.

    Returns:
        float: The p-value of the ADF test.
    """
    if not isinstance(series, (pd.Series, list, np.ndarray)):
        raise ValueError("Input series must be a pandas Series, list, or numpy array.")

    adf_result = adfuller(series)
    if isinstance(adf_result, tuple) and len(adf_result) > 1:
        return adf_result[1]  # P-value
    else:
        raise ValueError("Unexpected output from adfuller: {}".format(adf_result))  # Return the p-value of the ADF test

class TimeSeriesAnalysis:
    def __init__(self, df: pd.DataFrame, start_date: datetime, end_date: datetime, train_end: datetime,
                 test_end: datetime, column: str, date_column: str):
        """
        Initialize the TimeSeriesAnalysis class.

        Args:
            df (pd.DataFrame): The dataframe containing the time series data.
            start_date (datetime): The start date for analysis.
            end_date (datetime): The end date for analysis.
            train_end (datetime): The last date for training data.
            test_end (datetime): The last date for testing data.
            column (str): The target column for analysis.
            date_column (str): The column representing dates.
        """
        # Validate inputs with Pydantic
        validated_inputs = TimeSeriesAnalysisInput(df=df, start_date=start_date, end_date=end_date, train_end=train_end,
                                                   test_end=test_end, column=column, date_column=date_column)

        # Store validated inputs
        self.df = validated_inputs.df
        self.start_date = validated_inputs.start_date
        self.end_date = validated_inputs.end_date
        self.train_end = validated_inputs.train_end
        self.test_end = validated_inputs.test_end
        self.column = validated_inputs.column
        self.date_column = validated_inputs.date_column

        # Set the date column as index and ensure frequency
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df.set_index(date_column, inplace=True)
        self.df = self.df.asfreq('D')

    def check_stationarity(self, max_diffs: int = 2) -> Tuple[pd.Series, bool]:
        """
        Check if the series is stationary using the ADF test.

        Args:
            max_diffs (int): Maximum number of differencing attempts.

        Returns:
            Tuple[pd.Series, bool]: Differenced data and stationarity status.
        """
        p_value = adf_test(self.df[self.column])
        print(f'- ADF Test p-value (original): {p_value}')

        differenced_data = self.df[self.column]
        for i in range(max_diffs):
            if p_value > 0.05:
                differenced_data = differenced_data.diff().dropna()  # Apply differencing
                p_value = adf_test(differenced_data)
                print(f'- ADF Test p-value (after {i + 1} differencing): {p_value}\n ')
                if p_value <= 0.05:
                    print(f"Series is stationary after {i + 1} differencing -> d = {i + 1}.")
                    return differenced_data, True
            else:
                print(f"Series is stationary after {i} differencing -> d = {i}.")
                return differenced_data, True

        # If the loop ends, it means the series is still non-stationary after the max_diffs
        print(f"Series is still non-stationary after {max_diffs} differencing.")
        return differenced_data, False

    def plot_acf_pacf(self, acf_lag: int, pacf_lag: int) -> None:
        """
        Plot ACF and PACF for the time series.

        Args:
            acf_lag (int): Number of lags for the ACF plot.
            pacf_lag (int): Number of lags for the PACF plot.
        """
        plt.figure(figsize=(12, 6))

        # ACF plot
        plt.subplot(121)
        plot_acf(self.df[self.column], lags=acf_lag, ax=plt.gca()) #11
        plt.title("ACF Plot")

        # PACF plot
        plt.subplot(122)
        plot_pacf(self.df[self.column], lags=pacf_lag, ax=plt.gca()) #5
        plt.title("PACF Plot")
        plt.tight_layout()

    def plot_forecast(self, predictions: pd.Series, title: str) -> None:
        """
        Plot actual vs predicted values.

        Args:
            predictions (pd.Series): Predicted values.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.df[self.column], label='Actual')
        plt.plot(predictions, label='Predicted', color='orange')
        for month in pd.date_range(start=self.start_date, end=self.end_date, freq='MS'):
            plt.axvline(month, color='k', linestyle='--', alpha=0.2)
        plt.title(title)
        plt.ylabel('Flights')
        plt.legend()
        plt.show()

    def arima_sarimax_forecast(self, order: Tuple[int, int, int], seasonal_order: Optional[
        Tuple[int, int, int, int]] = None) -> Tuple[pd.Series, pd.Series, pd.Series, Union[ARIMA, SARIMAX]]:
        """
        Perform ARIMA or SARIMAX forecasting.

        Args:
            order (Tuple[int, int, int]): ARIMA order (p, d, q).
            seasonal_order (Optional[Tuple[int, int, int, int]]): SARIMAX seasonal order.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, Union[ARIMA, SARIMAX]]:
            Actual data, predictions, residuals, and model.
        """
        train_data = self.df[self.column][:self.train_end]
        test_data = self.df[self.column][self.train_end + timedelta(days=1):self.test_end]

        # Fit SARIMAX or ARIMA based on seasonal_order
        if seasonal_order:
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order).fit(disp=False)
            title = "SARIMAX Forecast vs Actual"
        else:
            model = ARIMA(train_data, order=order).fit()
            title = "ARIMA"

        print(model.summary())

        # Forecast
        forecast = model.get_forecast(steps=len(test_data))
        predictions = forecast.predicted_mean
        residuals = test_data - predictions

        # Plot results
        self.plot_forecast(predictions, title)

        # Evaluate
        return test_data, predictions, residuals, model

    def rolling_forecast(self, order: Tuple[int, int, int], train_window: int, forecast_steps: int = 1,
                         seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Perform rolling forecast.

        Args:
            order (Tuple[int, int, int]): ARIMA order.
            train_window (int): Number of observations in the training window.
            forecast_steps (int): Number of steps to forecast.
            seasonal_order (Optional[Tuple[int, int, int, int]]): SARIMAX seasonal order.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Actual data, predictions, and residuals.
        """
        rolling_predictions = []
        rolling_actual = []

        for i in range(train_window, len(self.df[self.column]) - forecast_steps + 1):
            train_data = self.df[self.column][:i]
            test_data = self.df[self.column][i:i + forecast_steps]

            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order).fit(disp=False)
            forecast = model.forecast(steps=forecast_steps)

            rolling_predictions.append(forecast.values[0])
            rolling_actual.append(test_data.values[0])

        rolling_predictions = pd.Series(rolling_predictions, index=self.df[self.column].index[train_window:len(
            self.df[self.column]) - forecast_steps + 1])
        rolling_actual = pd.Series(rolling_actual, index=self.df[self.column].index[train_window:len(
            self.df[self.column]) - forecast_steps + 1])

        # Calculate residuals
        residuals = rolling_actual - rolling_predictions

        # Plot results
        self.plot_forecast(rolling_predictions, title="Rolling Forecast vs Actual")
        return rolling_actual, rolling_predictions, residuals
