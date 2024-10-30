"""
general_utils.py - General purpose utility functions

This file contains functions that can be reused in various parts of the project.
These functions perform basic computations and data manipulations, such as calculating percentages and counting
specific conditions within DataFrames.
They can be used in different contexts within AvStats and beyond.
"""

import pandas as pd

def calc_percentage(part: int, whole: int) -> float:
    """
    Calculate the percentage of part over whole.

    Parameters:
    part (int): The numerator value.
    whole (int): The denominator value.

    Returns:
    float: The calculated percentage.
    """
    return (part / whole) * 100 if whole > 0 else 0

def count_delayed_flights(df: pd.DataFrame, lower: int, upper: int = None) -> int:
    """
    Count flights that are delayed within a specified range.

    Parameters:
    df (pd.DataFrame): The DataFrame containing flight data.
    lower (int): The lower bound for delay in minutes.
    upper (int, optional): The upper bound for delay in minutes. If not provided, no upper limit is applied.

    Returns:
    int: The number of delayed flights within the specified range.
    """
    if upper:
        return df[(df['dep_delay_15'] > lower) & (df['dep_delay_15'] <= upper)]['uuid'].count()
    return df[df['dep_delay_15'] > lower]['uuid'].count()