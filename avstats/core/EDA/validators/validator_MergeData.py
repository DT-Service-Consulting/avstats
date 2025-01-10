# core/EDA/validators/validator_MergeData.py
from pydantic import BaseModel, ConfigDict, field_validator, ValidationInfo
import pandas as pd
import numpy as np


class MergeDataInput(BaseModel):
    df: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("df")
    def validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {
            'sdt', 'uuid', 'route_iata_code', 'type', 'status', 'dep_delay', 'dep_delay_15',
            'on_time_15', 'dep_delay_cat', 'calc_sft', 'calc_aft', 'calc_flight_distance_km',
            'flight_cat', 'dep_time_window', 'arr_time_window', 'tavg_dep', 'prcp_dep',
            'snow_dep', 'wdir_dep', 'wspd_dep', 'wpgt_dep', 'pres_dep', 'tsun_dep',
            'tavg_arr', 'prcp_arr', 'snow_arr', 'wdir_arr', 'wspd_arr', 'wpgt_arr',
            'pres_arr', 'tsun_arr'
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"The DataFrame is missing the following required columns: {missing_columns}")
        return df
