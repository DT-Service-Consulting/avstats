

# New column flight_cat using the Flight class's categorize_flight method
def categorize_flight(row):
    flight = NewFeatures(uuid=row['uuid'], dep_delay=row['dep_delay'], sdt=row['sdt'], sat=row['sat'], cargo=row['cargo'], private=row['private'])
    return flight.categorize_flight()


# New column dep_time_window & arr_time_window using the Flight class's get_time_window method
def get_time_window_for_flight(row, time_type='departure'):
    flight = NewFeatures(uuid=row['uuid'], dep_delay=row['dep_delay'], sdt=row['sdt'], sat=row['sat'], cargo=row['cargo'], private=row['private'])
    return flight.get_time_window(time_type)
