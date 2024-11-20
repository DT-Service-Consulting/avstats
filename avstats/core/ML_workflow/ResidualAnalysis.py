# core/ML_workflow/ResidualAnalysis.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats

class ResidualAnalysis:
    def __init__(self, model, y_pred, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.residuals = y_test - self.y_pred

    def plot_residuals(self, dataset_name, subplot_position=None):
        """
        Plots residuals vs. predicted values in the specified subplot or as a standalone plot.

        Parameters:
        - subplot_position: tuple (nrows, ncols, index) for plt.subplot. If None, creates a standalone plot.
        """
        sns.set_theme(style="whitegrid")
        if subplot_position:
            plt.subplot(*subplot_position)  # Unpack the subplot position (nrows, ncols, index)
        else:
            plt.figure(figsize=(10, 6))

        plt.scatter(self.y_pred, self.residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')  # Add a line at zero
        plt.title(f'Errors vs Predicted Values - {dataset_name}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Errors')
        plt.xlim(0, 20000)
        plt.ylim(-max(abs(self.residuals)), max(abs(self.residuals)))
        plt.grid()

        # Show plot only if not in subplot mode
        if not subplot_position:
            plt.show()

    """
    def plot_histogram(self):
        # Histogram of residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(self.residuals, kde=True)
        plt.title('Residuals Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.show()

    def normality_test(self):
        # Perform normality test on residuals (Shapiro-Wilk test)
        stat, p_value = stats.shapiro(self.residuals)
        print(f"Shapiro-Wilk Test: Statistic={stat}, p-value={p_value}")
        return stat, p_value

    def homoscedasticity_test(self):
        # Plot residuals against fitted values for heteroscedasticity check
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_pred, y=self.residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values (Homoscedasticity Check)')
        plt.show()
    """