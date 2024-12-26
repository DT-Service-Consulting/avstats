"""
EDA_utils.py - Specified utility functions

This file contains utility functions specifically made for analyzing flight data as part of the AvStats project.
They focus on calculations related to flight performance metrics, including delay summaries, on-time performance,
and flight category percentages.
By separating these functions from those in the utils.py file, we improve the readability and maintainability of
the code, which allows for modifications to be made without affecting the general utility functions
"""

import pandas as pd


def calculate_time_window_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the total number and proportions of flights in each time window for departures and arrivals.

    Returns:
    pd.DataFrame: A DataFrame with time window and the corresponding departure and arrival percentages.
    """
    dep_time_window_counts = df['dep_time_window'].value_counts()
    arr_time_window_counts = df['arr_time_window'].value_counts()
    all_time_windows = dep_time_window_counts.index.union(arr_time_window_counts.index)
    dep_time_window_counts = dep_time_window_counts.reindex(all_time_windows, fill_value=0)
    arr_time_window_counts = arr_time_window_counts.reindex(all_time_windows, fill_value=0)
    dep_total_flights = dep_time_window_counts.sum()
    arr_total_flights = arr_time_window_counts.sum()
    dep_percentages = (dep_time_window_counts / dep_total_flights * 100).round(
        2) if dep_total_flights > 0 else pd.Series(dtype=float)
    arr_percentages = (arr_time_window_counts / arr_total_flights * 100).round(
        2) if arr_total_flights > 0 else pd.Series(dtype=float)

    return pd.DataFrame({
        'Time Window': all_time_windows,
        'Departure Percentages (%)': dep_percentages,
        'Arrival Percentages (%)': arr_percentages
    }).set_index('Time Window').fillna(0)  # Fill NaN values with 0 for categories that have no flights


def flight_summary_by_time_window(df: pd.DataFrame, time_window_col: str,
                                  summarize_delays: bool = False) -> pd.DataFrame:
    """
    Generate a flight summary by time window, optionally summarizing delay statistics.

    Parameters:
    time_window_col (str): The column name representing the time window (e.g., 'dep_time_window').
    summarize_delays (bool): Whether to summarize delayed flights (default is False).

    Returns:
    pd.DataFrame: A DataFrame with flight counts and delay percentages by time window.
    """
    total_flight_count = df.groupby(time_window_col).size().reset_index(
        name='total_flights')  # Count total flights by time window

    if summarize_delays:
        delayed_flights = df[df['dep_delay_15'] == 1].groupby(time_window_col).size().reset_index(
            name='delayed_flights')
        flight_summary = pd.merge(total_flight_count, delayed_flights, on=time_window_col, how='left').fillna(0)
        flight_summary[f'{time_window_col}_proportion_delayed'] = flight_summary['delayed_flights'] / \
                                                                  flight_summary['total_flights']
        flight_summary[f'{time_window_col}_percentage_delayed'] = ((flight_summary['delayed_flights'] /
                                                                    flight_summary['total_flights'] * 100).round(
            2)).astype(str) + '%'
    else:
        flight_summary = total_flight_count

    return flight_summary


def calculate_on_time_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate on-time performance for each flight category.

    Returns:
    pd.DataFrame: A DataFrame with the flight category, total flights, and on-time performance percentage.
    """
    total_flights = df['flight_cat'].value_counts().reset_index()  # Count total flights per category
    total_flights.columns = ['Flight Category', 'Total Flights']
    on_time_performance = df.groupby('flight_cat')[
        'on_time_15'].mean().reset_index()  # Calculate on-time performance for each flight type
    on_time_performance.columns = ['Flight Category', 'On-Time Performance']
    on_time_performance = pd.merge(on_time_performance, total_flights, on='Flight Category',
                                   how='left')  # Merge total flights with on-time performance
    on_time_performance['On-Time Performance'] = (on_time_performance['On-Time Performance'] * 100).round(2).astype(
        str) + '%'
    return on_time_performance[['Flight Category', 'Total Flights', 'On-Time Performance']]


def calculate_flight_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage of flights by category (Cargo, Commercial, Private).

    Returns:
    pd.DataFrame: A DataFrame with flight categories and their respective percentages.
    """
    total_flights = df['uuid'].count()
    cargo_flights = df[df['flight_cat'] == 'Cargo']['uuid'].count()
    commercial_flights = df[df['flight_cat'] == 'Commercial']['uuid'].count()
    private_flights = df[df['flight_cat'] == 'Private']['uuid'].count()
    cargo_percentage = ((cargo_flights / total_flights) * 100).round(2).astype(
        str) + '%' if total_flights > 0 else 0
    commercial_percentage = ((commercial_flights / total_flights) * 100).round(2).astype(
        str) + '%' if total_flights > 0 else 0
    private_percentage = ((private_flights / total_flights) * 100).round(2).astype(
        str) + '%' if total_flights > 0 else 0
    percentages_df = pd.DataFrame({
        'Flight Category': ['Cargo', 'Commercial', 'Private'],
        'Flight Amount Percentage': [cargo_percentage, commercial_percentage, private_percentage]
    })
    return percentages_df


def get_status_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the status of flights by counting occurrences and calculating proportions.

    Returns:
    pd.DataFrame: A DataFrame with flight status, total flights, and their proportions in percentages.
    """
    # Check if df is a valid DataFrame and has the 'status' column
    status_counts = df['status'].value_counts()  # Count the occurrences of each status type
    status_proportions = df['status'].value_counts(normalize=True) * 100  # Calculate the proportions (as percentages)
    rounded_proportions = status_proportions.round(2)
    status_summary = pd.DataFrame({
        'Status': status_counts.index,
        'Total Flights': status_counts.values,
        'Proportions (%)': rounded_proportions.values
    })
    return status_summary


def calculate_average_delay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average departure delay for each airline and route.

    Returns:
    pd.DataFrame: A DataFrame with airline name, route IATA code, and average departure delay.
    """
    average_delay = df.groupby(['airline_name', 'route_iata_code'])['dep_delay'].mean().round(2).reset_index()
    average_delay.rename(columns={'dep_delay': 'average_dep_delay'}, inplace=True)
    return average_delay.sort_values(by='average_dep_delay', ascending=False)
