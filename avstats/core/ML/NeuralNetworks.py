# core/ML/NeuralNetworks.py
import shap
import tensorflow as tf
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

    @staticmethod
    def explain_lstm(model, x_test):
        """
        Explain LSTM model predictions using SHAP.

        Args:
            model: Trained LSTM model.
            x_test: Test data used for prediction (3D array for LSTM).
        """
        import shap

        # Convert x_test to NumPy if it's a TensorFlow tensor
        if isinstance(x_test, tf.Tensor):
            x_test = x_test.numpy()

        # Ensure x_test has at least 10 samples
        num_samples = min(50, len(x_test))
        x_test_sample = x_test[:num_samples]

        # Check for proper dimensionality
        if len(x_test_sample.shape) != 3:
            raise ValueError(f"x_test_sample expected to have 3 dimensions, but got {x_test_sample.shape}")

        # Flatten x_test_sample to 2D for SHAP
        x_test_2d = x_test_sample.reshape(x_test_sample.shape[0], -1)

        # Define a function for SHAP that reshapes input back to LSTM format
        def model_predict(x):
            num_features = x_test_sample.shape[2]
            look_back = x_test_sample.shape[1]
            x_reshaped = x.reshape((x.shape[0], look_back, num_features))
            return model.predict(x_reshaped)

        # Initialize SHAP KernelExplainer
        explainer = shap.KernelExplainer(model_predict, x_test_2d)
        shap_values = explainer.shap_values(x_test_2d[:10])
        shap.summary_plot(shap_values, x_test_2d[:10])

    def neural_networks(self, x_standardized, y):
        look_back = self.look_back
        n_features = x_standardized.shape[1]
        n_samples = x_standardized.shape[0]

        # Ensure sufficient samples for the look-back window
        if n_samples <= look_back:
            raise ValueError(f"Insufficient samples ({n_samples}) for look-back window ({look_back}).")

        # Reshape the data dynamically
        x = np.array([x_standardized.values[i:i + look_back] for i in range(n_samples - look_back)])
        y = y[look_back:]  # Adjust target to align with input sequences

        # Split into train/test
        train_size = int(len(x) * 0.8)
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train model
        model = self.build_lstm_model((look_back, n_features))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test),
                  callbacks=[early_stopping])

        # Predictions
        predicted = model.predict(x_test)

        # Ensure y_test is 1D
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.squeeze()  # Convert to Series if it's a DataFrame
        y_test = y_test.values.flatten()  # Convert to a 1D NumPy array

        # Evaluation
        predicted = predicted.flatten() # Ensure predicted is flattened
        residuals = y_test - predicted # Calculate residuals

        print(f"Shape of y_test: {y_test.shape}")
        print(f"Shape of predicted: {predicted.shape}")
        print(f"First 5 residuals: {residuals[:5]}")

        # Evaluate the model
        metrics = evaluate_model(y_test, predicted, residuals)

        return model, x_test, y_test, predicted.flatten(), metrics


def nn_plots(axes, index, actual, predicted, title, metrics):
    """
    Plots actual vs. predicted values with metrics.

    Args:
        axes (matplotlib.axes.Axes or array-like): Axes to plot on.
        index (int): Index of the subplot (if axes is an array).
        actual (array-like): Actual values.
        predicted (array-like): Predicted values.
        title (str): Plot title.
        metrics (dict): Dictionary of evaluation metrics.
    """
    ax = axes if not isinstance(axes, (list, np.ndarray)) else axes[index]

    ax.plot(actual, label="Actual")
    ax.plot(predicted, label="Predicted")
    ax.set_title(f"{title}: Actual vs Predicted")
    ax.set_xlabel("Dataframe Index (Flights)")
    ax.set_ylabel("Delay (min.)")
    ax.legend()
    metrics_box(metrics, ax)
