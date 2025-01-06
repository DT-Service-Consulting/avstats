# core/ML_workflow/validators_ML/validator_ResidualAnalysis.py
from pydantic import BaseModel, ConfigDict, field_validator, ValidationInfo
import numpy as np


class ResidualAnalysisInput(BaseModel):
    model: str
    y_pred: np.ndarray
    residuals: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("y_pred", "residuals", mode="before")
    def validate_array_types(cls, value, info: ValidationInfo):
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{info.field_name} must be an instance of numpy.ndarray")
        if value.ndim != 1:
            raise ValueError(f"{info.field_name} must be a 1-dimensional numpy array")
        return value

    @field_validator("y_pred")
    def validate_y_pred_values(cls, y_pred):
        if np.any(y_pred < 0) or np.any(y_pred > 20000):
            raise ValueError("y_pred values must be in the range [0, 20000]")
        return y_pred

    @field_validator("residuals")
    def validate_lengths(cls, residuals, values: ValidationInfo):
        # Access y_pred from other fields
        y_pred = values.data.get("y_pred")
        if y_pred is not None and len(residuals) != len(y_pred):
            raise ValueError("y_pred and residuals must have the same length")
        return residuals

    @field_validator("y_pred", "residuals")
    def validate_numeric_values(cls, value, info: ValidationInfo):
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{info.field_name} must contain only numeric values")
        return value
