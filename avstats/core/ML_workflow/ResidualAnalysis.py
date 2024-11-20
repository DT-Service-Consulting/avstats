# core/ML_workflow/ResidualAnalysis.py
import matplotlib.pyplot as plt
import seaborn as sns
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
        - dataset_name (str): The name of the dataset being analyzed. This is used in the plot title.
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


    def q_q_normality(self, dataset_name, subplot_position=None):
        """
        Generates a Q-Q plot to check the normality of residuals.

        This method creates a Quantile-Quantile (Q-Q) plot to visually assess
        if the residuals from a dataset follow a normal distribution.

        Parameters:
        - dataset_name (str): The name of the dataset being analyzed. This is used in the plot title.
        - subplot_position (tuple, optional): The position of the subplot in the format
          (nrows, ncols, index). If provided, the plot will be part of a larger subplot grid.
          If not provided, the plot will be shown independently.
        """
        # Q-Q plot for normality check
        sns.set_theme(style="whitegrid")
        if subplot_position:
            plt.subplot(*subplot_position)  # Unpack the subplot position (nrows, ncols, index)
        else:
            plt.figure(figsize=(10, 6))

        stats.probplot(self.residuals, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of Residuals - {dataset_name}')

        # Show plot only if not in subplot mode
        if not subplot_position:
            plt.show()

    def histogram_normality(self, dataset_name, subplot_position=None):
        """
        Plots a histogram to visualize the distribution of residuals,
        helping to assess the normality of the residuals for a given dataset.

        Parameters:
        - dataset_name (str): The name of the dataset, used in the plot title.
        - subplot_position (tuple, optional): A tuple specifying the subplot position in the format
        (nrows, ncols, index). If provided, the histogram will be plotted in a subplot; otherwise, a
        new figure will be created.
        """
        sns.set_theme(style="whitegrid")
        if subplot_position:
            plt.subplot(*subplot_position)  # Unpack the subplot position (nrows, ncols, index)
        else:
            plt.figure(figsize=(10, 6))

        plt.hist(self.residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of Residuals - {dataset_name}')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.grid()

        # Show plot only if not in subplot mode
        if not subplot_position:
            plt.show()
