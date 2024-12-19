# NewFeatures.py


class NewFeatures:
    @staticmethod
    def categorize_flight(cargo: bool, private: bool) -> str:
        """
        Categorize the flight type as 'Cargo', 'Private', or 'Commercial'.

        Returns:
        str: The category of the flight.
        """
        if cargo:
            return 'Cargo'
        elif private:
            return 'Private'
        return 'Commercial'

    @staticmethod
    def get_time_window(hour: int) -> str:
        """
        Determine the time window (Morning, Afternoon, Evening) based on hour.

        Parameters:
        hour (int): Hour of the day.

        Returns:
        str: The time window.
        """
        if hour < 12:
            return 'Morning'
        elif hour < 18:
            return 'Afternoon'
        return 'Evening'

