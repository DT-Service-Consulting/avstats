### Table: `schedule_historical_2023`

| Column Name               | Data Type                  | Description                                                    |
| ------------------------- | -------------------------- | -------------------------------------------------------------- |
| `uuid`                    | `varchar`                  | Universally unique identifier for each record.                 |
| `route_iata_code`         | `varchar`                  | IATA code representing the route.                              |
| `type`                    | `varchar`                  | Type of schedule, can any of {arrival,departure}.              |
| `status`                  | `varchar`                  | Status of the flight (e.g., scheduled, departed, canceled).    |
| `dep_iata_code`           | `varchar`                  | IATA code of the departure airport.                            |
| `dep_icao_code`           | `varchar`                  | ICAO code of the departure airport.                            |
| `dep_delay`               | `integer`                  | Departure delay in minutes.                                    |
| `sdt`                     | `timestamp with time zone` | Scheduled Departure Time in UTC+00:00.                         |
| `adt`                     | `timestamp with time zone` | Actual Departure Time  in UTC+00:00.                           |
| `arr_iata_code`           | `varchar`                  | IATA code of the arrival airport.                              |
| `arr_icao_code`           | `varchar`                  | ICAO code of the arrival airport.                              |
| `sat`                     | `timestamp with time zone` | Scheduled Arrival Time in UTC+00:00.                           |
| `aat`                     | `timestamp with time zone` | Actual Arrival Time in UTC+00:00.                              |
| `airline_name`            | `varchar`                  | Name of the airline operating the flight.                      |
| `cargo`                   | `boolean = false`          | Indicates if the flight is a cargo flight. Default is `false`. |
| `private`                 | `boolean = false`          | Indicates if the flight is private. Default is `false`.        |
| `airline_iata_code`       | `varchar`                  | IATA code of the airline.                                      |
| `airline_icao_code`       | `varchar`                  | ICAO code of the airline.                                      |
| `flight_iata_number`      | `varchar`                  | IATA flight number.                                            |
| `flight_icao_number`      | `varchar`                  | ICAO flight number.                                            |
| `codeshared`              | `boolean = false`          | Indicates if the flight is codeshared. Default is `false`.     |
| `codeshared_iata_number`  | `varchar`                  | IATA number of the codeshared flight.                          |
| `calc_sft`                | `double precision`         | Calculated Scheduled Flight time in minutes (SAT-SDT).         |
| `calc_aft`                | `double precision`         | Calculated actual flight time in minutes (AAT-ADT).            |
| `calc_flight_distance_km` | `double precision`         | Calculated flight distance in kilometers.                      |


### Notes:

* **calc_flight_distance_km:** The calculated flight distance between airports is calculated using the [haversine](https://en.wikipedia.org/wiki/Haversine_formula) formula using the coordinates (Latitude & Longitude) of the departing and arrival city. It does not use the actual flight path of the plane

### Queries for dataset extraction
```
SELECT uuid,
route_iata_code,
type,
status,
dep_iata_code,
dep_delay,
sdt,
adt,
arr_iata_code,
sat,
aat,
airline_name,
cargo,
private,
airline_iata_code,
flight_iata_number,
calc_sft,
calc_aft,
calc_flight_distance_km
FROM schedule_historical_2023
WHERE route_iata_code like ('%CPH%')  -- all flights in and out of CPH Zaventem Airport
AND codeshared IS false -- remove codesharing flights
AND status NOT IN ('CANCELLED','UNKNOWN') -- get rid of flights without any possible delay
```

### Queries for Column inventory
**Count non-nulls and distinct values per column**
````
SELECT 
    COUNT(*) AS total_rows,

    COUNT(uuid) AS non_null_uuid,
    COUNT(DISTINCT uuid) AS distinct_uuid,
    
    COUNT(route_iata_code) AS non_null_route_iata_code,
    COUNT(DISTINCT route_iata_code) AS distinct_route_iata_code,
    
    COUNT(type) AS non_null_type,
    COUNT(DISTINCT type) AS distinct_type,
    
    COUNT(status) AS non_null_status,
    COUNT(DISTINCT status) AS distinct_status,
    
    COUNT(dep_iata_code) AS non_null_dep_iata_code,
    COUNT(DISTINCT dep_iata_code) AS distinct_dep_iata_code,
    
    COUNT(dep_icao_code) AS non_null_dep_icao_code,
    COUNT(DISTINCT dep_icao_code) AS distinct_dep_icao_code,
    
    COUNT(dep_delay) AS non_null_dep_delay,
    COUNT(DISTINCT dep_delay) AS distinct_dep_delay,
    
    COUNT(sdt) AS non_null_sdt,
    COUNT(DISTINCT sdt) AS distinct_sdt,
    
    COUNT(adt) AS non_null_adt,
    COUNT(DISTINCT adt) AS distinct_adt,
    
    COUNT(arr_iata_code) AS non_null_arr_iata_code,
    COUNT(DISTINCT arr_iata_code) AS distinct_arr_iata_code,
    
    COUNT(arr_icao_code) AS non_null_arr_icao_code,
    COUNT(DISTINCT arr_icao_code) AS distinct_arr_icao_code,
    
    COUNT(sat) AS non_null_sat,
    COUNT(DISTINCT sat) AS distinct_sat,
    
    COUNT(aat) AS non_null_aat,
    COUNT(DISTINCT aat) AS distinct_aat,
    
    COUNT(airline_name) AS non_null_airline_name,
    COUNT(DISTINCT airline_name) AS distinct_airline_name,
    
    COUNT(cargo) AS non_null_cargo,
    COUNT(DISTINCT cargo) AS distinct_cargo,
    
    COUNT(private) AS non_null_private,
    COUNT(DISTINCT private) AS distinct_private,
    
    COUNT(airline_iata_code) AS non_null_airline_iata_code,
    COUNT(DISTINCT airline_iata_code) AS distinct_airline_iata_code,
    
    COUNT(airline_icao_code) AS non_null_airline_icao_code,
    COUNT(DISTINCT airline_icao_code) AS distinct_airline_icao_code,
    
    COUNT(flight_iata_number) AS non_null_flight_iata_number,
    COUNT(DISTINCT flight_iata_number) AS distinct_flight_iata_number,
    
    COUNT(flight_icao_number) AS non_null_flight_icao_number,
    COUNT(DISTINCT flight_icao_number) AS distinct_flight_icao_number,
    
    COUNT(codeshared) AS non_null_codeshared,
    COUNT(DISTINCT codeshared) AS distinct_codeshared,
    
    COUNT(codeshared_iata_number) AS non_null_codeshared_iata_number,
    COUNT(DISTINCT codeshared_iata_number) AS distinct_codeshared_iata_number,
    
    COUNT(calc_sft) AS non_null_calc_sft,
    COUNT(DISTINCT calc_sft) AS distinct_calc_sft,
    
    COUNT(calc_aft) AS non_null_calc_aft,
    COUNT(DISTINCT calc_aft) AS distinct_calc_aft,
    
    COUNT(calc_flight_distance_km) AS non_null_calc_flight_distance_km,
    COUNT(DISTINCT calc_flight_distance_km) AS distinct_calc_flight_distance_km
FROM 
    schedule_historical_2023;
````

**Null values inventory **
   
````
SELECT 
    COUNT(*) - COUNT(uuid) AS null_uuid,
    COUNT(*) - COUNT(route_iata_code) AS null_route_iata_code,
    COUNT(*) - COUNT(type) AS null_type,
    COUNT(*) - COUNT(status) AS null_status,
    COUNT(*) - COUNT(dep_iata_code) AS null_dep_iata_code,
    COUNT(*) - COUNT(dep_icao_code) AS null_dep_icao_code,
    COUNT(*) - COUNT(dep_delay) AS null_dep_delay,
    COUNT(*) - COUNT(sdt) AS null_sdt,
    COUNT(*) - COUNT(adt) AS null_adt,
    COUNT(*) - COUNT(arr_iata_code) AS null_arr_iata_code,
    COUNT(*) - COUNT(arr_icao_code) AS null_arr_icao_code,
    COUNT(*) - COUNT(sat) AS null_sat,
    COUNT(*) - COUNT(aat) AS null_aat,
    COUNT(*) - COUNT(airline_name) AS null_airline_name,
    COUNT(*) - COUNT(cargo) AS null_cargo,
    COUNT(*) - COUNT(private) AS null_private,
    COUNT(*) - COUNT(airline_iata_code) AS null_airline_iata_code,
    COUNT(*) - COUNT(airline_icao_code) AS null_airline_icao_code,
    COUNT(*) - COUNT(flight_iata_number) AS null_flight_iata_number,
    COUNT(*) - COUNT(flight_icao_number) AS null_flight_icao_number,
    COUNT(*) - COUNT(codeshared) AS null_codeshared,
    COUNT(*) - COUNT(codeshared_iata_number) AS null_codeshared_iata_number,
    COUNT(*) - COUNT(calc_sft) AS null_calc_sft,
    COUNT(*) - COUNT(calc_aft) AS null_calc_aft,
    COUNT(*) - COUNT(calc_flight_distance_km) AS null_calc_flight_distance_km
FROM 
    schedule_historical_2023;
````

**Statistical Summary for Numeric and Date Columns**
````
SELECT 
    MIN(sdt) AS min_sdt,
    MAX(sdt) AS max_sdt,

    MIN(sdt) AS min_adt,
    MAX(sdt) AS max_adt,

    MIN(sdt) AS min_sat,
    MAX(sdt) AS max_sat,

    MIN(sdt) AS min_aat,
    MAX(sdt) AS max_aat,

    MIN(dep_delay) AS min_dep_delay,
    MAX(dep_delay) AS max_dep_delay,
    AVG(dep_delay) AS avg_dep_delay,
    
    MIN(calc_sft) AS min_calc_sft,
    MAX(calc_sft) AS max_calc_sft,
    AVG(calc_sft) AS avg_calc_sft,
    
    MIN(calc_aft) AS min_calc_aft,
    MAX(calc_aft) AS max_calc_aft,
    AVG(calc_aft) AS avg_calc_aft,
    
    MIN(calc_flight_distance_km) AS min_calc_flight_distance_km,
    MAX(calc_flight_distance_km) AS max_calc_flight_distance_km,
    AVG(calc_flight_distance_km) AS avg_calc_flight_distance_km
FROM 
    schedule_historical_2023;
````

**Boolean Column Inventory**
````
SELECT 
    SUM(CASE WHEN cargo = true THEN 1 ELSE 0 END) AS true_cargo,
    SUM(CASE WHEN cargo = false THEN 1 ELSE 0 END) AS false_cargo,
    
    SUM(CASE WHEN private = true THEN 1 ELSE 0 END) AS true_private,
    SUM(CASE WHEN private = false THEN 1 ELSE 0 END) AS false_private,
    
    SUM(CASE WHEN codeshared = true THEN 1 ELSE 0 END) AS true_codeshared,
    SUM(CASE WHEN codeshared = false THEN 1 ELSE 0 END) AS false_codeshared
FROM 
    schedule_historical_2023;
````