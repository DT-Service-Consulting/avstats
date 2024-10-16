from imports import *

class NewFeatures:
    def __init__(self, uuid, dep_delay, sdt, sat, cargo=False, private=False):
        self.uuid = uuid
        self.dep_delay = dep_delay
        self.sdt = pd.to_datetime(sdt)
        self.sat = pd.to_datetime(sat)
        self.cargo = cargo
        self.private = private

    def categorize_flight(self):
        if self.cargo:
            return 'Cargo'
        elif self.private:
            return 'Private'
        else:
            return 'Commercial'

    def get_time_window(self, time_type='departure'):
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
        """Calculate percentage."""
        return (part / whole) * 100 if whole > 0 else 0

    def count_delayed_flights(self, lower, upper=None):
        """Count delayed flights in a specified range."""
        if upper:
            return self.df[(self.df['dep_delay'] > lower) & (self.df['dep_delay'] <= upper)]['uuid'].count()
        return self.df[self.df['dep_delay'] > lower]['uuid'].count()

    def overall_performance(self):
        """Calculate overall flight performance percentages."""
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
        """Calculate percentage of delayed flights for specified ranges."""
        percentages = {}
        total_flights = self.df['uuid'].count()

        for lower, upper, label in delay_ranges:
            count = self.count_delayed_flights(lower, upper)
            percentages[label] = self.calc_percentage(count, total_flights)

        return percentages

    def calculate_time_window_percentages(self):
        """Total amount & Proportions of flights in each time window for departures and arrivals"""
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
        total_flights = self.df['flight_cat'].value_counts().reset_index() # Count total flights per category
        total_flights.columns = ['Flight Category', 'Total Flights']
        on_time_performance = self.df.groupby('flight_cat')['on_time_15'].mean().reset_index() # Calculate on-time performance for each flight type
        on_time_performance.columns = ['Flight Category', 'On-Time Performance']
        on_time_performance = pd.merge(on_time_performance, total_flights, on='Flight Category', how='left') # Merge total flights with on-time performance
        on_time_performance['On-Time Performance'] = (on_time_performance['On-Time Performance'] * 100).round(2).astype(str) + '%'

        return on_time_performance[['Flight Category', 'Total Flights', 'On-Time Performance']]

    def calculate_flight_percentages(self):
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
        """Group by airline_name and route_iata_code and calculate the mean of dep_delay"""
        average_delay = self.df.groupby(['airline_name', 'route_iata_code'])['dep_delay'].mean().round(2).reset_index()
        average_delay.rename(columns={'dep_delay': 'average_dep_delay'}, inplace=True)
        return average_delay.sort_values(by='average_dep_delay', ascending=False)

    def plot_top_delays(self, top_n=10):
        """Get the top N routes with the highest average delay times"""
        average_delay_sorted = self.calculate_average_delay()
        top_delays = average_delay_sorted.head(top_n) 

        plt.figure(figsize=(10, 6))
        sns.barplot(x='average_dep_delay', y='route_iata_code', hue='airline_name', data=top_delays, dodge=False)
        plt.title('Top Average Delays by Route and Airline')
        plt.xlabel('Average Delay (min.)')
        plt.ylabel('Route IATA Code')
        plt.show()
