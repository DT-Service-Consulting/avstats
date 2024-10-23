from imports import *

class DataCleaning:
    def __init__(self, unique_column):
        self.unique_column = unique_column  # Class variable to store the unique column name
        
    def check_missing_and_duplicates(self, df):
        """
        Function to check for missing values and duplicated rows in a DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to check.
        
        Returns:
        missing_values (int): The total number of missing values.
        duplicate_rows (pd.DataFrame): The DataFrame of duplicated rows.
        """
        # Check for missing values
        missing_values = df.isna().sum().sum()
        print(f"Number of total missing values: {missing_values}")
        
        # Check for duplicated rows based on the unique column
        duplicate_rows = df[df.duplicated(subset=self.unique_column, keep=False)]
        print(f"Number of total duplicated rows: {len(duplicate_rows)}")

        # Print missing values by column
        missing_by_column = df.isnull().sum()
        print("Missing values by column:")
        print(missing_by_column)

        return missing_values, duplicate_rows # Return missing values and duplicate rows DataFrame


class NewFeatures:
    def __init__(self, uuid, dep_delay, sdt, sat, cargo=False, private=False):
        self.uuid = uuid
        self.dep_delay = dep_delay
        self.sdt = pd.to_datetime(sdt)
        self.sat = pd.to_datetime(sat)
        self.cargo = cargo
        self.private = private

    def categorize_flight(self):
        """
        Categorize the flight type as 'Cargo', 'Private', or 'Commercial'.
        
        Returns:
        str: The category of the flight.
        """
        if self.cargo:
            return 'Cargo'
        elif self.private:
            return 'Private'
        else:
            return 'Commercial'

    def get_time_window(self, time_type='departure'):
        """
        Determine the time window of the flight (Morning, Afternoon, Evening).
        
        Parameters:
        time_type (str): Whether to calculate based on 'departure' or 'arrival'. Must be either 'departure' or 'arrival'.
                         
        Returns:
        str: The time window ('Morning', 'Afternoon', or 'Evening').
        
        Raises:
        ValueError: If time_type is not 'departure' or 'arrival'.
        """
        if time_type == 'departure':
            hour = self.sdt.hour
        elif time_type == 'arrival':
            hour = self.sat.hour
        else:
            raise ValueError("time_type must be either 'departure' or 'arrival'.")

        if hour < 12:
            return 'Morning'
        elif hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'
            

class FlightPerformance:
    def __init__(self, dataframe):
        self.df = dataframe

    def calc_percentage(self, part, whole):
        """
        Calculate the percentage of part over whole.
        
        Parameters:
        part (int): The numerator value.
        whole (int): The denominator value.
        
        Returns:
        float: The calculated percentage.
        """
        return (part / whole) * 100 if whole > 0 else 0

    def count_delayed_flights(self, lower, upper=None):
        """
        Count flights that are delayed within a specified range.
        
        Parameters:
        lower (int): The lower bound for delay in minutes.
        upper (int, optional): The upper bound for delay in minutes. If not provided, no upper limit is applied.
        
        Returns:
        int: The number of delayed flights within the specified range.
        """
        if upper:
            return self.df[(self.df['dep_delay'] > lower) & (self.df['dep_delay'] <= upper)]['uuid'].count()
        return self.df[self.df['dep_delay'] > lower]['uuid'].count()

    def overall_performance(self):
        """
        Calculate overall flight performance percentages (Delayed, On-Time, Missing Status).
        
        Returns:
        dict: A dictionary with performance categories and their respective percentages.
        """
        total_flights = self.df['uuid'].count()
        delayed_flights = self.df[self.df['dep_delay_15'] == 1]['uuid'].count()
        ontime_flights = self.df[self.df['on_time_15'] == 1]['uuid'].count()
        missing_status_flights = total_flights - (delayed_flights + ontime_flights)

        return {
            "Delayed Flights": self.calc_percentage(delayed_flights, total_flights),
            "On-Time Flights": self.calc_percentage(ontime_flights, total_flights),
            "Flights with Missing Status": self.calc_percentage(missing_status_flights, total_flights)
        }

    def delayed_flight_percentages(self, delay_ranges):
        """
        Calculate the percentage of delayed flights for specified ranges.
        
        Parameters:
        delay_ranges (list of tuples): A list of (lower, upper, label) defining the delay ranges.
        
        Returns:
        dict: A dictionary with delay ranges and their corresponding percentages.
        """
        percentages = {}
        total_flights = self.df['uuid'].count()
        for lower, upper, label in delay_ranges:
            count = self.count_delayed_flights(lower, upper)
            percentages[label] = self.calc_percentage(count, total_flights)
        return percentages

    def calculate_time_window_percentages(self):
        """
        Calculate the total number and proportions of flights in each time window for departures and arrivals.
        
        Returns:
        pd.DataFrame: A DataFrame with time window and the corresponding departure and arrival percentages.
        """
        dep_time_window_counts = self.df['dep_time_window'].value_counts()
        arr_time_window_counts = self.df['arr_time_window'].value_counts()
        all_time_windows = dep_time_window_counts.index.union(arr_time_window_counts.index)
        dep_time_window_counts = dep_time_window_counts.reindex(all_time_windows, fill_value=0)
        arr_time_window_counts = arr_time_window_counts.reindex(all_time_windows, fill_value=0)
        dep_total_flights = dep_time_window_counts.sum()
        arr_total_flights = arr_time_window_counts.sum()
        dep_percentages = (dep_time_window_counts / dep_total_flights * 100).round(2) if dep_total_flights > 0 else pd.Series(dtype=float)
        arr_percentages = (arr_time_window_counts / arr_total_flights * 100).round(2) if arr_total_flights > 0 else pd.Series(dtype=float)
        
        return pd.DataFrame({
            'Time Window': all_time_windows,
            'Departure Percentages (%)': dep_percentages,
            'Arrival Percentages (%)': arr_percentages
        }).set_index('Time Window').fillna(0)  # Fill NaN values with 0 for categories that have no flights
    
    def flight_summary_by_time_window(self, time_window_col, summarize_delays=False):
        """
        Generate a flight summary by time window, optionally summarizing delay statistics.
        
        Parameters:
        time_window_col (str): The column name representing the time window (e.g., 'dep_time_window').
        summarize_delays (bool): Whether to summarize delayed flights (default is False).
        
        Returns:
        pd.DataFrame: A DataFrame with flight counts and delay percentages by time window.
        """
        total_flight_count = self.df.groupby(time_window_col).size().reset_index(name='total_flights') # Count total flights by time window
        
        if summarize_delays:
            delayed_flights = self.df[self.df['dep_delay_15'] == 1].groupby(time_window_col).size().reset_index(name='delayed_flights')
            flight_summary = pd.merge(total_flight_count, delayed_flights, on=time_window_col, how='left').fillna(0)
            flight_summary[f'{time_window_col}_proportion_delayed'] = flight_summary['delayed_flights'] / flight_summary['total_flights']
            flight_summary[f'{time_window_col}_percentage_delayed'] = ((flight_summary['delayed_flights'] / flight_summary['total_flights'] * 100).round(2)).astype(str) + '%'
        else:
            flight_summary = total_flight_count
            
        return flight_summary

    def calculate_on_time_performance(self):
        """
        Calculate on-time performance for each flight category.
        
        Returns:
        pd.DataFrame: A DataFrame with the flight category, total flights, and on-time performance percentage.
        """
        total_flights = self.df['flight_cat'].value_counts().reset_index() # Count total flights per category
        total_flights.columns = ['Flight Category', 'Total Flights']
        on_time_performance = self.df.groupby('flight_cat')['on_time_15'].mean().reset_index() # Calculate on-time performance for each flight type
        on_time_performance.columns = ['Flight Category', 'On-Time Performance']
        on_time_performance = pd.merge(on_time_performance, total_flights, on='Flight Category', how='left') # Merge total flights with on-time performance
        on_time_performance['On-Time Performance'] = (on_time_performance['On-Time Performance'] * 100).round(2).astype(str) + '%'
        return on_time_performance[['Flight Category', 'Total Flights', 'On-Time Performance']]

    def calculate_flight_percentages(self):
        """
        Calculate the percentage of flights by category (Cargo, Commercial, Private).
        
        Returns:
        pd.DataFrame: A DataFrame with flight categories and their respective percentages.
        """
        total_flights = self.df['uuid'].count()
        cargo_flights = self.df[self.df['flight_cat'] == 'Cargo']['uuid'].count()
        commercial_flights = self.df[self.df['flight_cat'] == 'Commercial']['uuid'].count()
        private_flights = self.df[self.df['flight_cat'] == 'Private']['uuid'].count()
        cargo_percentage = ((cargo_flights / total_flights) * 100).round(2).astype(str) + '%' if total_flights > 0 else 0
        commercial_percentage = ((commercial_flights / total_flights) * 100).round(2).astype(str) + '%' if total_flights > 0 else 0
        private_percentage = ((private_flights / total_flights) * 100).round(2).astype(str) + '%' if total_flights > 0 else 0
        percentages_df = pd.DataFrame({
            'Flight Category': ['Cargo', 'Commercial', 'Private'],
            'Flight Amount Percentage': [cargo_percentage, commercial_percentage, private_percentage]
        })
        return percentages_df

    def get_status_summary(self):
        """
        Summarize the status of flights by counting occurrences and calculating proportions.
        
        Returns:
        pd.DataFrame: A DataFrame with flight status, total flights, and their proportions in percentages.
        """
        status_counts = self.df['status'].value_counts() # Count the occurrences of each status type
        status_proportions = self.df['status'].value_counts(normalize=True) * 100 # Calculate the proportions (as percentages)
        rounded_proportions = status_proportions.round(2)
        status_df = pd.DataFrame({
            'Status': status_counts.index,
            'Total Flights': status_counts.values,
            'Proportions (%)': rounded_proportions.values
        })
        return status_df


class AirlineDelays:
    def __init__(self, dataframe):
        self.df = dataframe

    def calculate_average_delay(self):
        """
        Calculate the average departure delay for each airline and route.
        
        Returns:
        pd.DataFrame: A DataFrame with airline name, route IATA code, and average departure delay.
        """
        average_delay = self.df.groupby(['airline_name', 'route_iata_code'])['dep_delay'].mean().round(2).reset_index()
        average_delay.rename(columns={'dep_delay': 'average_dep_delay'}, inplace=True)
        return average_delay.sort_values(by='average_dep_delay', ascending=False)

    def plot_top_delays(self, top_n=10):
        """
        Plot the top N routes with the highest average delay times.
        
        Parameters:
        top_n (int): The number of top delays to plot (default is 10).
        """
        average_delay_sorted = self.calculate_average_delay()
        top_delays = average_delay_sorted.head(top_n) 
        plt.figure(figsize=(10, 6))
        sns.barplot(x='average_dep_delay', y='route_iata_code', hue='airline_name', data=top_delays, dodge=False)
        plt.title('Top Average Delays by Route and Airline')
        plt.xlabel('Average Delay (min.)')
        plt.ylabel('Route IATA Code')
        plt.show()


class Multicollinearity:
    def __init__(self, dataframe):
        self.df = dataframe
        
    def calculate_vif(self):
        """
        Calculate Variance Inflation Factor (VIF) for each feature to assess multicollinearity.
        
        Returns:
        pd.DataFrame: A DataFrame with features and their corresponding VIF values.
        """
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.df.columns
        vif_data["VIF"] = [variance_inflation_factor(self.df.values, i) for i in range(self.df.shape[1])]
        return vif_data
    
    def remove_high_vif_features(self, target_variable, threshold=10):
        """
        Iteratively remove features with high VIF values until all remaining features have VIF below a threshold.
        
        Parameters:
        target_variable (str): The target variable to exclude from VIF calculation.
        threshold (float): The VIF threshold above which features will be removed (default is 10).
        
        Returns:
        pd.DataFrame: A DataFrame with the target variable and features with VIF below the threshold.
        """
        features = self.df.columns[self.df.columns != target_variable] # Exclude the target variable from the features
    
        while True:
            vif_data = pd.DataFrame() # Calculate VIF for the current features
            vif_data["feature"] = features
            vif_data["VIF"] = [variance_inflation_factor(self.df[features].values, i) for i in range(len(features))]
            high_vif_features = vif_data[vif_data["VIF"] > threshold] # Identify features with VIF above the threshold
            
            if high_vif_features.empty:
                print("No features above the VIF threshold.")
                break
            
            feature_to_remove = high_vif_features.sort_values("VIF", ascending=False).iloc[0]["feature"] # Remove the feature with the highest VIF
            print(f"Removing feature: {feature_to_remove}")
            features = features[features != feature_to_remove]
    
        return self.df[[target_variable] + list(features)] # Return the cleaned DataFrame including the target variable
