# core/ML_workflow/TimeSeriesAnalysis.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test and return the p-value.
    """
    adf_result = adfuller(series)
    return adf_result[1]  # Return the p-value of the ADF test

class TimeSeriesAnalysis:
    def __init__(self, df, start_date, end_date, train_end, test_end, column, date_column):
        self.df = df
        self.column = column
        self.date_column = date_column
        self.train_end = train_end
        self.test_end = test_end
        self.start_date = start_date
        self.end_date = end_date

        # Ensure the date column is a datetime object
        #self.df[date_column] = pd.to_datetime(self.df[date_column])

        # Set the date column as the index and ensure frequency
        #self.df.set_index(date_column, inplace=True)
        self.df = self.df.asfreq('D')

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

    def plot_forecast(self, predictions, title):
        """Plot actual vs predicted values with monthly vertical lines."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.df[self.column], label='Actual')
        plt.plot(predictions, label='Predicted', color='orange')
        for month in pd.date_range(start=self.start_date, end=self.end_date, freq='MS'):
            plt.axvline(month, color='k', linestyle='--', alpha=0.2)
        plt.title(title)
        plt.ylabel('Flights')
        plt.legend()
        plt.show()

    def arima_sarimax_forecast(self, order, seasonal_order=None):
        """
       Combined ARIMA and SARIMAX forecasting function.
       If seasonal_order is None, ARIMA will be used. Otherwise, SARIMAX will be used.
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

    def rolling_forecast(self, order, train_window, forecast_steps=1, seasonal_order=None):
        """Perform rolling forecast using SARIMAX."""
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


class NeuralNetworks:
    def __init__(self, df):
        self.df = df

    def neural_networks(self):
        # Load and prepare the data
        values = self.df['total_dep_delay'].values  # Replace with your column name
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values.reshape(-1, 1))

        # Create sliding windows
        def create_dataset(data, look_back=1):
            x, y = [], []
            for i in range(len(data) - look_back - 1):
                x.append(data[i:(i + look_back), 0])
                y.append(data[i + look_back, 0])
            return np.array(x), np.array(y)

        look_back = 10  # Number of past days used for prediction
        x, y = create_dataset(scaled_values, look_back)

        # Reshape for LSTM input
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        # Train-test split
        train_size = int(len(x) * 0.8)
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build the LSTM model with an explicit Input layer
        model = Sequential([
            Input(shape=(look_back, 1)),  # Define the input shape explicitly
            LSTM(50),
            Dropout(0.2),
            Dense(1)  # Single output for regression
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1)

        # Predictions
        predicted = model.predict(x_test)
        nn_predictions = scaler.inverse_transform(predicted)  # Inverse scaling
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot actual vs predicted
        plt.plot(y_test, label="Actual")
        plt.plot(nn_predictions, label="Predicted")
        plt.legend()
        plt.show()
        return y_test, nn_predictions