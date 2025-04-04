# core/ML/NeuralNetworks.py
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Tuple
from avstats.core.ML.ModelEvaluation import *
from avstats.core.ML.ModelEvaluation import evaluate_model, metrics_box


class NeuralNetworks:
    def __init__(self, df: pd.DataFrame, column: str, look_back=10):
        self.df = df
        self.column = column
        self.look_back = look_back

    @staticmethod
    def create_dataset(data: np.ndarray, lookback: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        x_nn, y_nn = [], []
        for i in range(len(data) - lookback - 1):
            x_nn.append(data[i:(i + lookback), 0])
            y_nn.append(data[i + lookback, 0])
        return np.array(x_nn), np.array(y_nn)

    @staticmethod
    def build_lstm_model(input_shape):
        model = Sequential([Input(shape=input_shape), LSTM(50), Dropout(0.2), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        return model

    def neural_networks(self, dataset_name) -> tuple:
        """
        Trains an LSTM model, evaluates its performance, and stores results in a structured format.

        Args:
            dataset_name (str): Name of the dataset for identification.

        Returns:
            dict: Model summary containing metrics, predictions, and configuration details.
        """
        model_summaries = []

        # Scale data
        values = self.df[self.column].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values.reshape(-1, 1))

        # Create dataset
        x, y = self.create_dataset(scaled_values, self.look_back)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        # Split data
        train_size = int(len(x) * 0.8)
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train model
        model = self.build_lstm_model((self.look_back, 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])

        # Predictions
        predicted = model.predict(x_test)
        predictions_inverse = scaler.inverse_transform(predicted)
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Evaluation
        residuals = y_test_inverse - predictions_inverse
        metrics = evaluate_model(y_test_inverse, predictions_inverse, residuals)

        # Store results
        model_summary = {
            "Dataset": dataset_name,
            "Model Type": "LSTM",
            "Look Back Period": self.look_back,
            "Final Validation Loss": history.history["val_loss"][-1] if "val_loss" in history.history else None,
            "MAE (min.)": metrics["MAE (min.)"],
            "MAPE (%)": metrics["MAPE (%)"],
            "RMSE (min.)": metrics["RMSE (min.)"],
            "Total Training Epochs": len(history.history["loss"]),
        }
        model_summaries.append(model_summary)

        return model, y_test_inverse, predictions_inverse, model_summary


def nn_plots(axes, index, actual, predicted, title): #, metrics):
    """
    Plots actual vs. predicted values with metrics.

    Args:
        axes (matplotlib.axes.Axes or array-like): Axes to plot on.
        index (int): Index of the subplot (if axes is an array).
        actual (array-like): Actual values.
        predicted (array-like): Predicted values.
        title (str): Plot title.
    """
    ax = axes if not isinstance(axes, (list, np.ndarray)) else axes[index]

    ax.plot(actual, label="Actual")
    ax.plot(predicted, label="Predicted")
    ax.set_title(f"{title}: Actual vs Predicted")
    ax.set_xlabel("Dataframe Index (Flights)")
    ax.set_ylabel("Delay (min.)")
    ax.legend()
    # metrics_box(metrics, ax)
