# For all visualization-related functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics(evaluation_results):
    df = pd.DataFrame(evaluation_results).set_index("Model")
    df[['MAE (min.)', 'RMSE (min.)']].plot(kind='bar', figsize=(12, 4), alpha=0.7)
    plt.title('Model Performance (MAE and RMSE)')
    plt.ylabel('(min.)')
    plt.xlabel('')
    plt.xticks(rotation=0, horizontalalignment='center')  # Rotate labels for better readability
    plt.legend(title="Metrics")
    plt.show()

    df[['MAPE (%)']].plot(kind='bar', figsize=(12, 4), alpha=0.7)
    plt.title('Model Performance (MAPE)')
    plt.ylabel('(%)')
    plt.xlabel('')
    plt.ylim(0, 100)  # Set y-axis range from 0 to 100
    plt.xticks(rotation=0, horizontalalignment='center')  # Rotate labels for better readability
    plt.legend(title="Metrics")
    plt.show()
