# core/ML/TimeSeriesAnalysis.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
from numpy import ndarray, dtype
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from datetime import timedelta, datetime
from typing import Tuple, Optional, List, Union, Any
from avstats.core.ML.ModelEvaluation import *
from avstats.core.ML.validators.validator_TimeSeriesAnalysis import TimeSeriesAnalysisInput


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
                 test_end: datetime, column: str):
        """
        Initialize the TimeSeriesAnalysis class.

        Args:
            df (pd.DataFrame): The dataframe containing the time series data.
            start_date (datetime): The start date for analysis.
            end_date (datetime): The end date for analysis.
            train_end (datetime): The last date for training data.
            test_end (datetime): The last date for testing data.
            column (str): The target column for analysis.
        """
        # Validate inputs with Pydantic
        validated_inputs = TimeSeriesAnalysisInput(df=df, start_date=start_date, end_date=end_date, train_end=train_end,
                                                   test_end=test_end, column=column)

        # Set Seaborn style globally
        sns.set_style("whitegrid")

        # Store validated inputs
        self.df = validated_inputs.df
        self.start_date = validated_inputs.start_date
        self.end_date = validated_inputs.end_date
        self.train_end = validated_inputs.train_end
        self.test_end = validated_inputs.test_end
        self.column = validated_inputs.column

        # Convert 'Date' to datetime and set as index
        self.prophet_df = None
        self.df = self.df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)

        # Resample and forward-fill missing values
        self.df = self.df.resample('D').ffill()
        self.df = self.df.loc[start_date:end_date]

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

    def plot_forecast(self, predictions: pd.Series, title: str, metrics, prophet_column=None, prophet_plot=False) -> None:
        """
        Plot actual vs predicted values.

        Args:
            prophet_plot:
            prophet_column:
            predictions (pd.Series): Predicted values.
            title (str): Title of the plot.
            metrics (dict): Evaluation metrics to display.
        """
        plt.figure(figsize=(12, 4))
        if prophet_plot:
            plt.plot(self.prophet_df['ds'], self.prophet_df[prophet_column], label='Actual')
            plt.plot(predictions.index, predictions, label='Predicted', color='orange')
            plt.xticks(rotation=45)

            # Format the x-axis with proper dates
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        else:
            plt.plot(self.df[self.column], label='Actual')
            plt.plot(predictions, label='Predicted', color='orange')
            plt.xticks(rotation=45)
            for month in pd.date_range(start=self.start_date, end=self.end_date, freq='MS'):
                plt.axvline(month, color='k', linestyle='--', alpha=0.2)
        plt.title(title, fontsize=14)
        plt.ylabel('Flights')
        metrics_box(metrics)
        plt.legend()
        plt.show()

    def plot_combined(self, model_name, predictions: pd.Series, residuals=None):
        """
        Plot actual vs predicted values and residuals side by side.

        Args:
            model_name (str): Name of the model.
            predictions (pd.Series): Predicted values.
            residuals (pd.Series, optional): Residuals (actual - predicted). Defaults to None.
        """
        actual = self.df[self.column]  # Get actual values
        actual = actual.loc[predictions.index]  # Align actual values with prediction index
        residuals = residuals if residuals is not None else actual - predictions

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Ensure x-axis alignment using the index
        actual_dates = actual.index
        prediction_dates = predictions.index
        residual_dates = residuals.index  # Ensure residuals match the x-axis

        # Actual vs Predicted plot
        axes[0].plot(actual_dates, actual, label='Actual')
        axes[0].plot(prediction_dates, predictions, label='Predicted', color='orange')
        axes[0].set_title(f'{model_name}: Actual vs Predicted', fontsize=14)

        # Ensure vertical lines for test periods are on the correct axis
        for month in pd.date_range(start=self.train_end, end=self.test_end, freq='MS'):
            axes[0].axvline(month, color='k', linestyle='--', alpha=0.2)

        axes[0].legend()
        axes[0].tick_params(axis="x", rotation=45, labelsize=10)

        # Residuals plot
        axes[1].plot(residual_dates, residuals, label='Residuals', color='purple')
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.7)
        axes[1].set_title(f'{model_name}: Residuals', fontsize=14)
        axes[1].legend()
        axes[1].tick_params(axis="x", rotation=45, labelsize=10)

        plt.tight_layout()
        plt.show()

    def arima_sarimax_forecast(self, order: Tuple[int, int, int], seasonal_order: Optional[
    Tuple[int, int, int, int]] = None) -> tuple[ARIMAResults | Any, Any, Any, Any, dict[str, float | None]]:
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
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                            enforce_invertibility=False).fit(disp=False, maxiter=500, method="powell")
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
        metrics = evaluate_model(test_data, predictions, residuals)
        self.plot_forecast(predictions, title, metrics)

        # Evaluate
        return model, test_data, predictions, residuals, metrics

    def rolling_forecast(self, order: Tuple[int, int, int], train_window: int, forecast_steps: int = 1,
                         seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> tuple[
        ndarray[Any, dtype[Any]], Series, Series, dict[str, float | None]]:
        """
        Perform rolling forecast.

        Args:
        order (Tuple[int, int, int]): ARIMA order.
        train_window (int): Number of observations in the training window.
        forecast_steps (int): Number of steps to forecast.
        seasonal_order (Optional[Tuple[int, int, int, int]]): SARIMAX seasonal order.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Actual data, predictions, and residuals.
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

        # Convert actual and predictions to NumPy arrays (to match expected function return type)
        rolling_actual_array = np.array(rolling_actual)
        rolling_predictions_array = np.array(rolling_predictions)

        # Generate index for Pandas Series to align with timestamps
        index_range = self.df[self.column].index[train_window: len(self.df[self.column]) - forecast_steps + 1]

        # Convert residuals to Pandas Series for plotting
        rolling_residuals_series = pd.Series(rolling_actual_array - rolling_predictions_array, index=index_range)

        # Convert to NumPy arrays before evaluation (expected format)
        metrics = evaluate_model(rolling_actual_array, rolling_predictions_array, rolling_residuals_series.values)

        # Plot results using Series format
        rolling_predictions_series = pd.Series(rolling_predictions_array, index=index_range)
        self.plot_forecast(rolling_predictions_series, title="Rolling Forecast vs Actual", metrics=metrics)

        return rolling_actual_array, rolling_predictions_series, rolling_residuals_series, metrics

    def prophet_forecast(self, periods, frequency='D'):
        """
        Perform forecasting using Facebook Prophet.

        Args:
            periods (int): Number of periods to forecast into the future.
            frequency (str): Frequency of the time series ('D' for daily, 'M' for monthly, etc.).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
            - Forecast dataframe (including future dates and predictions).
            - Actual vs predicted dataframe.
            - Evaluation metrics (MAE, RMSE).
        """
        # Preparing dataset
        self.prophet_df = self.df.copy()
        self.prophet_df = self.prophet_df.resample('D').ffill()
        self.prophet_df.reset_index(inplace=True)
        self.prophet_df.rename(columns={"Date": "ds", "total_dep_delay": "y"}, inplace=True)

        # Debugging: Print date range of prophet_data
        print("Prophet data range:", self.prophet_df['ds'].min(), "-", self.prophet_df['ds'].max())

        # Prepare the training and test data
        train_data = self.prophet_df[(self.prophet_df['ds'] <= self.train_end)]
        test_data = self.prophet_df[(self.prophet_df['ds'] > self.train_end) & (self.prophet_df['ds'] <= self.test_end)]

        # Debugging: Check if train and test datasets have data
        print(f"Train data range: {train_data['ds'].min()} - {train_data['ds'].max()} ({len(train_data)} rows)")
        print(f"Test data range: {test_data['ds'].min()} - {test_data['ds'].max()} ({len(test_data)} rows)")

        # Check for empty train or test data
        if train_data.empty:
            raise ValueError("Train data is empty. Check your train_end date.")
        if test_data.empty:
            raise ValueError("Test data is empty. Check your test_end date.")

        # Initialize Prophet
        model = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=10)
        model.add_seasonality(name='monthly', period=30, fourier_order=3)
        model.fit(train_data)

        # Generate future dataframe
        future = model.make_future_dataframe(periods=periods, freq=frequency)
        forecast = model.predict(future)

        # Extract predictions for the test period
        predictions = forecast[['ds', 'yhat']].set_index('ds').reindex(test_data['ds'], fill_value=np.nan)

        # Evaluate predictions
        actual_vs_predicted = pd.DataFrame({
            "Actual": test_data.set_index('ds')['y'],
            "Predicted": predictions['yhat']
        })

        # Metrics
        actual_vs_predicted.dropna(inplace=True)
        predictions_metrics = actual_vs_predicted['Predicted']
        residuals = actual_vs_predicted['Actual'] - predictions_metrics
        metrics = evaluate_model(actual_vs_predicted['Actual'], predictions_metrics, residuals)

        # Plot results
        self.plot_forecast( predictions_metrics, "Prophet Forecast vs Actual", metrics, "y", prophet_plot=True)

        return forecast, actual_vs_predicted, metrics
