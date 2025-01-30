# Visualizations.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import pi


def annotate_bars(ax, data, use_enumeration=False, offset=0, label_format="{:.2f}", color='white', ha='right'):
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
            ax.text(v + offset, i, label_format.format(v, p), color=color, ha=ha, va='center', fontsize=10)
    else:
        for i, v in enumerate(data):
            ax.text(v + offset, i, label_format.format(v), color=color, ha=ha, va='center', fontsize=10)

def plot_radar_and_flight_cat(df):
    """
    Combines a radar chart and a bar plot into a single figure.

    Parameters:
    df (pd.DataFrame): Flight dataset with required columns.
    """
    # Extract day of the week and month
    df['DayOfWeek'] = pd.to_datetime(df['sdt']).dt.day_name()
    df['Month'] = pd.to_datetime(df['sdt']).dt.to_period('M')

    # Flight Type
    cat_counts = df['flight_cat'].value_counts()
    cat_counts_percentage = (cat_counts / df['flight_cat'].count() * 100).round(2)

    # Radar chart data
    radar_data = {
        'Routes': df.groupby('route_iata_code')['dep_delay'].mean().head(10).mean(),
        'Distance': df.groupby('calc_flight_distance_km')['dep_delay'].mean().mean(),
        'Flight Type': df.groupby('flight_cat')['dep_delay'].mean().mean(),
        'Time of Day': df.groupby('dep_time_window')['dep_delay'].mean().mean(),
        'Day of Week': df.groupby('DayOfWeek')['dep_delay'].mean().mean(),
        'Month': df.groupby('Month')['dep_delay'].mean().mean(),
    }
    labels = list(radar_data.keys())
    values = list(radar_data.values())
    values += values[:1]  # Close the radar chart
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    plt.figure(figsize=(16, 4))
    sns.set_style("whitegrid")

    # Radar chart
    plt.subplot(121, polar=True)
    plt.xticks(angles[:-1], labels, color='grey', size=9)
    plt.plot(angles, values, linewidth=2, linestyle='solid', label='Average Delay', color='purple')
    plt.fill(angles, values, alpha=0.4, color='purple')
    plt.title('Characteristics of Delayed Flights', size=14, pad=20)
    plt.legend(loc='center', bbox_to_anchor=(1.3, 1.1))

    # Barplot
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="viridis", ax=ax2)
    plt.title("Flight Category Distribution", size=14, pad=20)
    plt.xlabel("Total Flights")
    plt.ylabel("")
    annotate_bars(ax2, zip(cat_counts.values, cat_counts_percentage), use_enumeration=True, offset=5000,
                  label_format="{:,} flights ({:.2f}%)", color='black', ha='left')
    plt.tight_layout()
    plt.show()

def plot_dep_delay_distribution(df, delay_column='dep_delay'):
    """
    Plots the distribution of departure delays with a histogram and a boxplot.

    Parameters:
    df (pd.DataFrame): The flight data DataFrame.
    delay_column (str): Column name for the departure delay.
    """
    # Filter delays within a reasonable range for better visualization
    filtered_df = df[(df[delay_column] >= -100) & (df[delay_column] <= 300)]
    plt.figure(figsize=(16, 4))
    sns.set_style("whitegrid")

    # Subplot 1: Histogram with KDE for filtered delay distribution
    plt.subplot(1, 2, 1)
    sns.histplot(filtered_df[delay_column], bins=50, kde=True, color='darkgreen')
    plt.title("Distribution of Departure Delays (Filtered)", fontsize=14)
    plt.xlabel("Departure Delay (min.)")
    plt.ylabel("Amount of Flights")

    # Subplot 2: Boxplot for detecting outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=filtered_df[delay_column], color='darkorange')
    plt.title("Boxplot of Departure Delays (Filtered)", fontsize=14)
    plt.xlabel("Departure Delay (min.)")
    plt.tight_layout()
    plt.show()

    # Plot of extreme outliers
    outliers_df = df[(df[delay_column] < -100) | (df[delay_column] > 300)]
    if not outliers_df.empty:
        plt.figure(figsize=(16, 3))
        sns.histplot(outliers_df[delay_column], bins=50, kde=False, color='darkred')
        plt.title("Extreme Outliers in Departure Delays", fontsize=14)
        plt.xlabel("Departure Delay (minutes)")
        plt.ylabel("Amount of Flights")
        plt.tight_layout()
        plt.show()

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

    # Least Common Routes
    least_common_routes = df[route_column].value_counts(ascending=True).head(top_n)

    # Calculate Average Delay by Route
    average_delay_sorted = df.groupby(route_column)[delay_column].mean().sort_values(ascending=False).reset_index()
    average_delay_sorted.columns = [route_column, 'average_delay']

    # Subplot 1: Most Common Routes
    plt.figure(figsize=(16, 4))
    sns.set_style("whitegrid")
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(x=common_routes.values, y=common_routes.index, palette='Oranges_d', ax=ax1)
    plt.title('Top 10 Most Common Routes')
    plt.xlabel('Total Flights')
    plt.ylabel('')
    annotate_bars(ax1, zip(common_routes.values, common_routes_percentage), use_enumeration=True, offset=-500, label_format="{:,} flights ({:.2f}%)")
    plt.xlim(0, 7000)
    plt.tight_layout()

    # Subplot 2: Least Common Routes
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(x=least_common_routes.values, y=least_common_routes.index, palette='Greens_d', ax=ax2)
    plt.title('Top 10 Least Common Routes')
    plt.xlabel('Total Flights')
    plt.ylabel('')
    plt.xlim(0, 7)
    plt.tight_layout()
    plt.show()

    """
    # Subplot 2: Routes with Highest Delays
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(data=average_delay_sorted.head(top_n), x='average_delay', y=route_column, palette='Oranges_d', ax=ax2)
    plt.title('Top 10 Routes with Highest Delays')
    plt.xlabel('Average Delay (min.)')
    plt.ylabel('')
    annotate_bars(ax2, average_delay_sorted.head(top_n)['average_delay'].values, use_enumeration=False, offset=-5, label_format="{:.2f} min.")
    plt.tight_layout()
    plt.show()"""

    # Boxplot of Delays for Top Routes
    top_routes = df[route_column].value_counts().nlargest(top_n).index
    filtered_df = df[df[route_column].isin(top_routes)]
    plt.figure(figsize=(16, 4))
    sns.boxplot(data=filtered_df, x=route_column, y=delay_column, color='red')
    plt.title('Delay Distribution by Top 10 Routes')
    plt.xlabel('')
    plt.ylabel('Delay (min.)')
    plt.xticks(rotation=45)
    plt.ylim(-100, 500)
    plt.tight_layout()
    plt.show()

def plot_distance_vs_delay(df, distance_column='calc_flight_distance_km', delay_column='dep_delay'):
    plt.figure(figsize=(16, 4))
    sns.set_style("whitegrid")
    sns.regplot(data=df, x=distance_column, y=delay_column, scatter_kws={'s': 10, 'alpha': 0.5}, line_kws={'color': 'red'})
    plt.title("Flight Distance vs. Departure Delay", fontsize=14)
    plt.xlabel("Flight Distance (km)")
    plt.ylabel("Departure Delay (min.)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

def plot_airports_with_delays(df, delay_column='dep_delay', top_n=10):
    """
    Plots the top airports prone to delays based on average delay time.

    Parameters:
    df (pd.DataFrame): The flight data DataFrame.
    delay_column (str): Column name for the delay values.
    top_n (int): Number of top airports to display in the plot.
    """
    # Group by the airport column and calculate the mean delay
    dep_airport_delays = df.groupby('dep_iata_code')[delay_column].mean().sort_values(ascending=False).head(top_n)
    arr_airport_delays = df.groupby('arr_iata_code')[delay_column].mean().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(16, 4))
    sns.set_style("whitegrid")

    # Create a bar plot for departure airports
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(x=dep_airport_delays.values, y=dep_airport_delays.index, palette='BuPu_d', ax=ax1)
    ax1.set_title(f"Top {top_n} Departure Airports Prone to Delays", fontsize=14)
    ax1.set_xlabel("Average Delay (min.)")
    ax1.set_ylabel("Airport")
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    annotate_bars(ax1, dep_airport_delays.values, offset=1, label_format="{:.2f} min.", color='black', ha='left')

    # Create a bar plot for arrival airports
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(x=arr_airport_delays.values, y=arr_airport_delays.index, palette='PuBu_d', ax=ax2)
    ax2.set_title(f"Top {top_n} Arrival Airports Prone to Delays", fontsize=14)
    ax2.set_xlabel("Average Delay (min.)")
    ax2.set_ylabel("Airport")
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    annotate_bars(ax2, arr_airport_delays.values, offset=1, label_format="{:.2f} min.", color='black', ha='left')
    plt.tight_layout()
    plt.show()

def plot_airline_avg_delays(df, delay_column='dep_delay', airline_column='airline_name', top_n=10):
    # Calculate average delays by airline
    airline_delays = df.groupby(airline_column)[delay_column].mean().sort_values()

    plt.figure(figsize=(16, 6))
    sns.set_style("whitegrid")

    # Top airlines with the highest average delays
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(x=airline_delays.tail(top_n).values, y=airline_delays.tail(top_n).index, palette='Reds_d', ax=ax1)
    ax1.set_title(f"Top {top_n} Airlines with Highest Average Delays", fontsize=14)
    ax1.set_xlabel("Average Delay (min.)")
    ax1.set_ylabel("Airline")
    for i, (value, airline) in enumerate(zip(airline_delays.tail(top_n).values, airline_delays.tail(top_n).index)):
        ax1.text(value + 1, i, f"{value:.2f} min.", color='black', va='center', fontsize=10)

    # Top airlines with the lowest average delays
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(x=airline_delays.head(top_n).values, y=airline_delays.head(top_n).index, palette='Greens_d', ax=ax2)
    ax2.set_title(f"Top {top_n} Airlines with Lowest Average Delays", fontsize=14)
    ax2.set_xlabel("Average Delay (min.)")
    ax2.set_ylabel("Airline")
    for i, (value, airline) in enumerate(zip(airline_delays.head(top_n).values, airline_delays.head(top_n).index)):
        ax2.text(value + 1, i, f"{value:.2f} min.", color='black', va='center', fontsize=10)

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
    plt.figure(figsize=(16, 6))
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

def plot_scheduled_time_vs_delay(df, time_column='sdt', delay_column='dep_delay'):
    """
    Plots the relationship between scheduled departure time and departure delays.

    Parameters:
    df (pd.DataFrame): The flight data DataFrame.
    time_column (str): Column name for the scheduled departure time.
    delay_column (str): Column name for the departure delay.
    """
    # Convert scheduled time to datetime and extract the hour
    df['ScheduledHour'] = pd.to_datetime(df[time_column]).dt.hour

    # Group by hour and calculate the average delay
    hourly_delays = df.groupby('ScheduledHour')[delay_column].mean().reset_index()

    # Plot the relationship
    plt.figure(figsize=(16, 4))
    sns.lineplot(data=hourly_delays, x='ScheduledHour', y=delay_column, marker='o')
    plt.title("Relationship Between Scheduled Departure Time and Delays", fontsize=14)
    plt.xlabel("Scheduled Departure Hour (24-hour format)")
    plt.ylabel("Average Delay (min.)")
    plt.xticks(range(0, 24))
    plt.ylim(0, 50)
    plt.xlim(-1, 24)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for x, y in zip(hourly_delays['ScheduledHour'], hourly_delays[delay_column]):
        plt.text(x, y + 2.5, f"{y:.2f}", color='black', ha='center', fontsize=10)
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

def plot_weekly_and_monthly_comparison(df, delay_column='dep_delay', palette=None):
    """
    Plot combined analysis for delays and flight counts by day of the week and month.

    Parameters:
    df (pd.DataFrame): The flight data DataFrame.
    delay_column (str): The column containing delay values.
    palette (str): Color palette for the bar plots.
    """
    # Compute average delays and total flights by day of the week
    weekday_delays = df.groupby('DayOfWeek')[delay_column].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    weekday_flight_counts = df['DayOfWeek'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Compute average delays and total flights by month
    monthly_delays = df.groupby('Month')[delay_column].mean().sort_index()
    monthly_flight_counts = df['Month'].value_counts().sort_index()

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Average delays by day of the week
    sns.barplot(x=weekday_delays.index, y=weekday_delays.values, palette=palette, ax=axes[0, 0])
    axes[0, 0].set_title('Average Delays by Day of the Week')
    axes[0, 0].set_ylabel('Average Delay (min.)')
    for index, value in enumerate(weekday_delays.values):
        axes[0, 0].text(index, value - 3, f"{value:.2f} min.", color='white', ha='center', fontsize=10)
    axes[0, 0].set_ylim(0, 40)

    # Plot 2: Total flights by day of the week
    sns.barplot(x=weekday_flight_counts.index, y=weekday_flight_counts.values, palette=palette, ax=axes[0, 1])
    axes[0, 1].set_title('Total Flights by Day of the Week')
    axes[0, 1].set_ylabel('Total Flights')
    for index, value in enumerate(weekday_flight_counts.values):
        axes[0, 1].text(index, value - 3500, f"{value:,}", color='white', ha='center', fontsize=10)
    axes[0, 1].set_ylim(0, weekday_flight_counts.max() + 20000)

    # Plot 3: Average delays by month
    sns.lineplot(x=monthly_delays.index.astype(str), y=monthly_delays.values, marker='o', ax=axes[1, 0])
    axes[1, 0].set_title('Average Delays by Month')
    axes[1, 0].set_ylabel('Average Delay (min.)')
    axes[1, 0].set_xticklabels(monthly_delays.index.astype(str), rotation=45)
    axes[1, 0].set_ylim(0, 50)
    for x, y in zip(monthly_delays.index.astype(str), monthly_delays.values):
        axes[1, 0].text(x, y + 1.5, f"{y:.2f}", color='black', ha='center', fontsize=10)

    # Plot 4: Total flights by month
    sns.lineplot(x=monthly_flight_counts.index.astype(str), y=monthly_flight_counts.values, marker='o', ax=axes[1, 1])
    axes[1, 1].set_title('Total Flights by Month')
    axes[1, 1].set_ylabel('Total Flights')
    axes[1, 1].set_xticklabels(monthly_flight_counts.index.astype(str), rotation=45)
    axes[1, 1].set_ylim(0, 55000)
    for x, y in zip(monthly_flight_counts.index.astype(str), monthly_flight_counts.values):
        axes[1, 1].text(x, y + 2000, f"{y:,}", color='black', ha='center', fontsize=10)

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
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

    plt.figure(figsize=(16, 3))
    ax = sns.barplot(x=performance_values, y=performance_labels, palette=palette)
    plt.xlabel("Percentage (%)")
    plt.title("Overall Flight Performance")
    plt.xlim(0, 100)
    annotate_bars(ax=ax, data=performance_values, use_enumeration=False, offset=6, label_format="{:.2f}%",
                  color='black')
    plt.tight_layout()
    plt.show()
