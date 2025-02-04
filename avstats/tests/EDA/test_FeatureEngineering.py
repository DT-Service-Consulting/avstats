import pytest
from avstats.core.EDA.FeatureEngineering import FeatureEngineering


class TestFeatureEngineering:
    @pytest.mark.parametrize("cargo, private, expected", [
        (True, False, 'Cargo'),
        (False, True, 'Private'),
        (False, False, 'Commercial'),
        (True, True, 'Cargo')  # Assuming cargo takes precedence
    ])
    def test_categorize_flight(self, cargo, private, expected):
        assert FeatureEngineering.categorize_flight(cargo, private) == expected

    @pytest.mark.parametrize("hour, expected", [
        (0, 'Morning'),
        (6, 'Morning'),
        (11, 'Morning'),
        (12, 'Afternoon'),
        (15, 'Afternoon'),
        (17, 'Afternoon'),
        (18, 'Evening'),
        (21, 'Evening'),
        (23, 'Evening')
    ])
    def test_get_time_window(self, hour, expected):
        assert FeatureEngineering.get_time_window(hour) == expected

    def test_get_time_window_invalid_hour(self):
        with pytest.raises(ValueError):
            FeatureEngineering.get_time_window(-1)
        with pytest.raises(ValueError):
            FeatureEngineering.get_time_window(24)
