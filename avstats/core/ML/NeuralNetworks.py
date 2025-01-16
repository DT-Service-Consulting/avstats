# core/ML/NeuralNetworks.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from avstats.core.ML.validators.validator_NeuralNetworks import NeuralNetworksInput
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


class NeuralNetworks:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the NeuralNetworks class.

        Args:
            df (pd.DataFrame): The dataframe containing the time series data.
        """
        # Validate input using the Pydantic model
        validated_input = NeuralNetworksInput(df=df)
        self.df = validated_input.df

    @staticmethod
    def create_dataset(data: np.ndarray, lookback: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create datasets for time series modeling using sliding windows.

        Args:
            data (np.ndarray): Scaled time series data.
            lookback (int): Number of past observations to use for predicting the next value.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix (x) and target vector (y).
        """
        x_nn, y_nn = [], []
        for i in range(len(data) - lookback - 1):
            x_nn.append(data[i:(i + lookback), 0])
            y_nn.append(data[i + lookback, 0])
        return np.array(x_nn), np.array(y_nn)

    def neural_networks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build and train a neural network model (LSTM) for time series prediction.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Actual values (y_test) and predicted values (nn_predictions).
        """
        values = self.df['total_dep_delay'].values  # Replace with your column name
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values.reshape(-1, 1))

        look_back = 10  # Number of past days used for prediction
        x, y = self.create_dataset(scaled_values, look_back)

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