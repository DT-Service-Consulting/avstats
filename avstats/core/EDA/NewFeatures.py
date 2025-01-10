# NewFeatures.py
from avstats.core.EDA.validators.validator_NewFeatures import CategorizeFlightInput, GetTimeWindowInput


class NewFeatures:
    @staticmethod
    def categorize_flight(cargo: bool, private: bool) -> str:
        """
        Categorize the flight type as 'Cargo', 'Private', or 'Commercial'.

        Returns:
        str: The category of the flight.
        """
        # Validate inputs using Pydantic
        validated_input = CategorizeFlightInput(cargo=cargo, private=private)

        if validated_input.cargo:
            return 'Cargo'
        elif validated_input.private:
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

        Raises:
        ValueError: If the hour is not between 0 and 23.
        """
        # Validate inputs using Pydantic
        validated_input = GetTimeWindowInput(hour=hour)

        if validated_input.hour < 0 or validated_input.hour > 23:
            raise ValueError("Hour must be between 0 and 23")
        if validated_input.hour < 12:
            return 'Morning'
        elif validated_input.hour < 18:
            return 'Afternoon'
        return 'Evening'
