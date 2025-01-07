# core/ML/validators/validator_ResidualAnalysis.py
from pydantic import BaseModel, ConfigDict, field_validator, ValidationInfo
import numpy as np
import pandas as pd
from typing import Union, Any


class ResidualAnalysisInput(BaseModel):
    model: Any  # Allow any type for the model
    y_pred: np.ndarray
    residuals: Union[np.ndarray, pd.Series]  # Accept both numpy.ndarray and pandas.Series

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("y_pred", "residuals", mode="before")
    def validate_numpy_array(cls, value, info: ValidationInfo):
        if isinstance(value, pd.Series):
            return value.to_numpy()
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{info.field_name} must be an instance of numpy.ndarray or pandas.Series")
        return value

    @field_validator("y_pred")
    def validate_y_pred_range(cls, y_pred):
        if not np.all((0 <= y_pred) & (y_pred <= 20000)):
            raise ValueError("y_pred values must be in the range [0, 20000]")
        return y_pred

    @field_validator("residuals")
    def validate_lengths(cls, residuals, values: ValidationInfo):
        y_pred = values.data.get("y_pred")
        if y_pred is not None and len(residuals) != len(y_pred):
            raise ValueError("y_pred and residuals must have the same length")
        return residuals
