# core/ML/NeuralNetworks.py
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
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

    def neural_networks(self) -> tuple:
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
        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])

        # Predictions
        predicted = model.predict(x_test)
        predictions_inverse = scaler.inverse_transform(predicted)
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Evaluation
        residuals = y_test_inverse - predictions_inverse
        metrics = evaluate_model(y_test_inverse, predictions_inverse, residuals)

        return model, y_test_inverse, predictions_inverse, metrics

def metric_box(evaluation_metrics, ax):
    """
    Add a metrics box to a specific axis.

    Args:
        evaluation_metrics (dict): A dictionary of evaluation metrics.
        ax (matplotlib.axes._subplots.AxesSubplot): The subplot axis to add the metrics box to.
    """
    metrics_text = "\n\n".join([f"{key}: {value:.2f}" for key, value in evaluation_metrics.items()])
    props = dict(boxstyle="round,pad=0.4", edgecolor="gray", facecolor="whitesmoke")
    ax.text(
        1.05, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', horizontalalignment='left', bbox=props
    )

def nn_plots(axes, index, actual, predicted, title, metrics):
    axes[index].plot(actual, label="Actual")
    axes[index].plot(predicted, label="Predicted")
    axes[index].set_title(f"{title}: Actual vs Predicted")
    axes[index].set_xlabel("Dataframe Index (Flights)")
    axes[index].set_ylabel("Delay (min.)")
    axes[index].legend()
    metric_box(metrics, axes[index])
