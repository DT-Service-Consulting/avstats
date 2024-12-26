import pytest
import pandas as pd
from avstats.core.EDA_workflow.MergeData import MergeData

@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    data = {
        'uuid': [1, 2, 3, 4],
        'route_iata_code': ['AA101', 'AA101', 'BB202', 'BB202'],
        'type': ['DEPARTURE', 'ARRIVAL', 'DEPARTURE', 'ARRIVAL'],
        'status': ['LANDED', 'ACTIVE', 'SCHEDULED', 'LANDED'],
        'dep_delay': [5, 0, 10, None],
        'dep_delay_15': [1, 0, 0, 0],
        'on_time_15': [0, 1, 1, 0],
        'dep_delay_cat': ['Short', None, 'Medium', None],
        'calc_sft': [30, 20, 40, 50],
        'calc_aft': [60, 50, 40, 30],
        'calc_flight_distance_km': [500, 600, 700, 800],
        'flight_cat': ['Commercial', 'Private', 'Cargo', 'Commercial'],
        'dep_time_window': ['Morning', 'Afternoon', 'Evening', None],
        'arr_time_window': ['Morning', None, 'Evening', 'Afternoon'],
        'tavg_dep': [15.0, 20.0, 10.0, None],
        'prcp_dep': [0.1, 0.0, 0.2, None],
        'snow_dep': [0.0, 0.0, 0.0, None],
        'wdir_dep': [180, 200, 190, None],
        'wspd_dep': [5.0, 10.0, 15.0, None],
        'wpgt_dep': [2.0, 3.0, 1.0, None],
        'pres_dep': [1015, 1013, 1012, None],
        'tsun_dep': [100, 200, 300, None],
        'tavg_arr': [16.0, 19.0, 11.0, None],
        'prcp_arr': [0.0, 0.2, 0.1, None],
        'snow_arr': [0.0, 0.0, 0.1, None],
        'wdir_arr': [185, 195, 180, None],
        'wspd_arr': [6.0, 12.0, 14.0, None],
        'wpgt_arr': [2.5, 3.5, 1.5, None],
        'pres_arr': [1014, 1012, 1011, None],
        'tsun_arr': [110, 210, 310, None],
        'sdt': ['2023-12-01T08:00:00', '2023-12-01T12:00:00', '2023-12-02T18:00:00', '2023-12-02T20:00:00']
    }
    return pd.DataFrame(data)

def test_preprocess_datetime(sample_data):
    """Test the preprocess_datetime method."""
    merge_data = MergeData(sample_data)
    merge_data.preprocess_datetime('sdt')

    assert 'Date' in merge_data.df.columns
    assert pd.to_datetime(merge_data.df['sdt'], errors='coerce').notna().all()
    assert merge_data.df['Date'].dtype == 'object'

def test_aggregate_daily(sample_data):
    """Test the aggregate_daily method."""
    merge_data = MergeData(sample_data)
    merge_data.preprocess_datetime('sdt')
    aggregated_df = merge_data.aggregate_daily()

    assert isinstance(aggregated_df, pd.DataFrame)
    assert 'route_iata_code' in aggregated_df.columns
    assert 'Month' in aggregated_df.columns
    assert 'total_flights' in aggregated_df.columns

    # Check specific aggregations
    assert aggregated_df.loc[aggregated_df['route_iata_code'] == 'AA101', 'total_flights'].iloc[0] == 2
    assert aggregated_df.loc[aggregated_df['route_iata_code'] == 'BB202', 'total_flights'].iloc[0] == 2
    assert aggregated_df.loc[aggregated_df['route_iata_code'] == 'AA101', 'departures'].iloc[0] == 1
    assert aggregated_df.loc[aggregated_df['route_iata_code'] == 'BB202', 'total_flight_distance_km'].iloc[0] == 1500

def test_aggregate_passengers(sample_data):
    """Test the aggregate_passengers method."""
    merge_data = MergeData(sample_data)
    merge_data.preprocess_datetime('sdt')
    aggregated_df = merge_data.aggregate_daily()

    # Create a mock passengers DataFrame
    df_passengers = pd.DataFrame({
        'route_code': ['AA101', 'BB202'],
        '2023-12': [100, 200]
    })

    merged_df = merge_data.aggregate_passengers(df_passengers)

    assert isinstance(merged_df, pd.DataFrame)
    assert 'route_iata_code' in merged_df.columns
    assert 'total_passengers' in merged_df.columns
    assert merged_df.loc[merged_df['route_iata_code'] == 'AA101', 'total_passengers'].iloc[0] == 100
    assert merged_df.loc[merged_df['route_iata_code'] == 'BB202', 'total_passengers'].iloc[0] == 200
