# Visualizations.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def annotate_bars(ax, data, use_enumeration=False, offset=0, label_format="{:.2f}", color='white'):
    """
    Annotates bars in a bar plot with text.

    Parameters:
    ax (matplotlib.axes.Axes): The axis object for the plot.
    data (iterable): The data to annotate. If `use_enumeration` is True, it should be a zipped iterable of values and percentages.
    use_enumeration (bool): Whether to enumerate the data (used for paired values like count and percentage).
    offset (float): Offset for the text annotations.
    label_format (str): Format string for the label.
    """
    if use_enumeration:
        for i, (v, p) in enumerate(data):
            ax.text(v + offset, i, label_format.format(v, p), color=color, ha='right', va='center', fontsize=10)
    else:
        for i, v in enumerate(data):
            ax.text(v + offset, i, label_format.format(v), color=color, ha='right', va='center', fontsize=10)


def plot_route_analysis(df, route_column='route_iata_code', delay_column='dep_delay', top_n=10):
    """
    Generalized function to analyze and plot route data including:
    1. Most common routes.
    2. Routes with the highest average delays.
    3. Boxplot of delays for top routes.

    Parameters:
    df (pd.DataFrame): The flight data DataFrame.
    route_column (str): Column name for the route identifier.
    delay_column (str): Column name for the delay values.
    top_n (int): Number of top routes to display in the analysis.
    """
    # Most Common Routes
    common_routes = df[route_column].value_counts().head(top_n)
    common_routes_percentage = (common_routes / df[route_column].count() * 100).round(2)

    # Calculate Average Delay by Route
    average_delay_sorted = df.groupby(route_column)[delay_column].mean().sort_values(ascending=False).reset_index()
    average_delay_sorted.columns = [route_column, 'average_delay']

    # Subplot 1: Most Common Routes
    plt.figure(figsize=(12, 4))
    sns.set_style("whitegrid")
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(x=common_routes.values, y=common_routes.index, palette='Blues_d', ax=ax1)
    plt.title('Top 10 Most Common Routes')
    plt.xlabel('Total Flights')
    plt.ylabel('')
    annotate_bars(ax1, zip(common_routes.values, common_routes_percentage), use_enumeration=True, offset=-500, label_format="{:,} flights ({:.2f}%)")
    plt.tight_layout()

    # Subplot 2: Routes with Highest Delays
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(data=average_delay_sorted.head(top_n), x='average_delay', y=route_column, palette='Reds_d', ax=ax2)
    plt.title('Top 10 Routes with Highest Delays')
    plt.xlabel('Average Delay (min.)')
    plt.ylabel('')
    annotate_bars(ax2, average_delay_sorted.head(top_n)['average_delay'].values, use_enumeration=False, offset=-5, label_format="{:.2f} min.")
    plt.tight_layout()
    plt.show()

    # Boxplot of Delays for Top Routes
    top_routes = df[route_column].value_counts().nlargest(top_n).index
    filtered_df = df[df[route_column].isin(top_routes)]
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=filtered_df, x=route_column, y=delay_column, palette='Reds_d')
    plt.title('Delay Distribution by Top 10 Routes')
    plt.xlabel('')
    plt.ylabel('Delay (min.)')
    plt.xticks(rotation=45)
    plt.ylim(-100, 500)
    plt.tight_layout()
    plt.show()


def plot_precipitation_impact(df_weather, precipitation_column='prcp_dep', delay_column='dep_delay'):
    """
    Analyzes and visualizes the impact of precipitation on delays.

    Parameters:
    df_weather (pd.DataFrame): DataFrame containing weather and delay data.
    precipitation_column (str): Column name for precipitation levels.
    delay_column (str): Column name for delays.
    """
    bins = [-0.1, 0, 10, 30, float('inf')]
    delay_labels = ['On-Time', 'Moderate', 'Severe']
    labels = ['No Precipitation', 'Light Rain', 'Moderate Rain', 'Heavy Rain']

    df_weather_plot = df_weather.copy()

    # Categorize precipitation levels
    df_weather_plot['precipitation_level'] = pd.cut(df_weather_plot[precipitation_column], bins=bins, labels=labels, include_lowest=True)
    df_weather_plot['delay_category'] = pd.cut(df_weather_plot[delay_column], bins=[-float('inf'), 15, 60, float('inf')], labels=delay_labels)

    print("Distribution of Precipitation Levels:")
    print(df_weather_plot['precipitation_level'].value_counts())

    # Scatter plot of precipitation vs. delay
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_weather_plot, x=precipitation_column, y=delay_column, hue='precipitation_level', palette='viridis', alpha=0.7)
    plt.title('Scatter Plot of Precipitation vs. Departure Delay')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Delay (min.)')
    plt.legend(title='Precipitation Level')
    plt.xlim(-5, 100)
    plt.ylim(-1000, 2000)
    plt.tight_layout()

    # Bar plot of delay categories by precipitation level
    plt.subplot(1, 2, 2)
    sns.countplot(data=df_weather_plot, x='delay_category', hue='precipitation_level', palette='viridis')
    plt.title('Delay Categories by Precipitation Level')
    plt.xlabel('Delay Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_time_window(df, time_window_column='dep_time_window', delay_column='dep_delay', volume_palette=None, delay_palette=None):
    """
    Analyzes and visualizes flight volume and delays by time window.

    Parameters:
    df (pd.DataFrame): DataFrame containing flight data.
    time_window_column (str): Column name for time window categorization (e.g., morning, afternoon).
    delay_column (str): Column name for delay values.
    volume_palette (str): Color palette for the flight volume plot.
    delay_palette (str): Color palette for the delay plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))

    # Flight Volume by Time of Day
    flight_summary = df.groupby(time_window_column)['uuid'].count()
    flight_summary_percentage = (flight_summary / df[time_window_column].count() * 100).round(2)
    ax1 = sns.barplot(x=flight_summary.values, y=flight_summary.index, palette=volume_palette, ax=axes[0])
    axes[0].set_title('Flight Volume by Time Window')
    axes[0].set_xlabel('Amount of Flights')
    axes[0].set_ylabel('')
    annotate_bars(ax1, zip(flight_summary.values, flight_summary_percentage), use_enumeration=True, offset=-3000,
                  label_format="{:,} flights ({:.2f}%)")

    # Delays by Time of Day
    time_of_day_delays = df.groupby(time_window_column)[delay_column].mean()
    ax2 = sns.barplot(x=time_of_day_delays.values, y=time_of_day_delays.index, palette=delay_palette, ax=axes[1])
    axes[1].set_title('Average Departure Delays by Time Window')
    axes[1].set_xlabel('Delay (min.)')
    axes[1].set_ylabel('')
    annotate_bars(ax2, time_of_day_delays.values, use_enumeration=False, offset=-1, label_format="{:.2f} min.")

    plt.tight_layout()
    plt.show()

def plot_weekly_and_monthly(df, day_column='sdt', delay_column='dep_delay', palette=None):
    """
    Plot combined analysis for delays by day of the week and month.

    Parameters:
    df (pd.DataFrame): The flight data DataFrame.
    day_column (str): The column containing datetime information.
    delay_column (str): The column containing delay values.
    """
    # Create delays by day of the week
    df['DayOfWeek'] = pd.to_datetime(df[day_column]).dt.day_name()
    weekday_delays = df.groupby('DayOfWeek')[delay_column].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Create delays by month
    df['Month'] = pd.to_datetime(df[day_column]).dt.to_period('M')
    monthly_delays = df.groupby('Month')[delay_column].mean()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Delays by Day of the Week
    sns.barplot(x=weekday_delays.index, y=weekday_delays.values, palette=palette, ax=axes[0])
    axes[0].set_title('Average Delays by Day of the Week')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Average Delay (min.)')
    for index, value in enumerate(weekday_delays.values):
        axes[0].text(index, value - 3, f"{value:.2f} min.", color='white', ha='center', fontsize=10)
    axes[0].set_ylim(0, 40)

    # Plot 2: Delays by Month
    sns.lineplot(x=monthly_delays.index.astype(str), y=monthly_delays.values, marker='o', ax=axes[1])
    axes[1].set_title('Average Delays by Month')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Average Delay (min.)')
    axes[1].set_xticklabels(monthly_delays.index.astype(str), rotation=45)
    axes[1].set_ylim(0, 60)
    for x, y in zip(monthly_delays.index.astype(str), monthly_delays.values):
        axes[1].text(x, y + 2, f"{y:.2f}", color='black', ha='center', fontsize=10)

    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.show()

def plot_overall_performance(performance_metrics, palette="Blues_d"):
    """
    Plot overall flight performance as a bar chart.

    Parameters:
    performance_metrics (dict): Dictionary containing performance metrics and their percentages.
    title (str): Title for the chart.
    palette (str): Color palette for the bar chart.
    """
    performance_labels = list(performance_metrics.keys())
    performance_values = list(performance_metrics.values())

    plt.figure(figsize=(10, 3))
    ax = sns.barplot(x=performance_values, y=performance_labels, palette=palette)
    plt.xlabel("Percentage (%)")
    plt.title("Overall Flight Performance")
    plt.xlim(0, 100)
    annotate_bars(ax=ax, data=performance_values, use_enumeration=False, offset=6, label_format="{:.2f}%",
                  color='black')
    plt.tight_layout()
    plt.show()