# core/ML_workflow/ARIMA.py
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test and return the p-value.
    """
    adf_result = adfuller(series)
    return adf_result[1]  # Return the p-value of the ADF test

class TimeSeriesAnalysis:
    def __init__(self, df, column, date_column):
        self.df = df
        self.column = column
        self.date_column = date_column
        self.model = None

        # Ensure the date column is a datetime object
        self.df[date_column] = pd.to_datetime(self.df[date_column])

        # Set the date column as the index and ensure frequency
        self.df.set_index(date_column, inplace=True)
        self.df = self.df.asfreq('MS')  # Monthly Start Frequency (example)

    def check_stationarity(self, max_diffs=2):
        """
        Check if the series is stationary using the ADF test.
        If not stationary, apply differencing up to a specified number of times (max_diffs).
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

    def plot_acf_pacf(self, acf_lag, pacf_lag):
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

    def fit_arima(self, order):
        # Fit the ARIMA model
        self.model = ARIMA(self.df[self.column], order=order).fit(method_kwargs={"maxiter": 500, "disp": False})

        return self.model

    def plot_forecast(self, steps):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call 'fit_arima' first.")

        # Generate forecast
        forecast = self.model.get_forecast(steps=steps)
        conf_int = forecast.conf_int()

        # Generate forecast index
        forecast_index = pd.date_range(
            start=self.df.index[-1] + pd.offsets.MonthBegin(1),
            periods=steps,
            freq=self.df.index.freq
        )

        # Plot actual and forecasted values
        plt.figure(figsize=(10, 6))
        plt.plot(self.df.index, self.df[self.column], label="Actual")
        plt.plot(forecast_index, forecast.predicted_mean, label="Forecast", linestyle='--')
        plt.fill_between(
            forecast_index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color='gray',
            alpha=0.2,
            label="Confidence Interval"
        )
        plt.title(f"Forecast for {self.column}")
        plt.legend()
        plt.show()