# core/ML_workflow/validators_ML/validator_model_evaluation.py
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationInfo
import numpy as np
from typing import Optional


class CrossValidationInput(BaseModel):
    x_train: np.ndarray
    y_train: np.ndarray
    cv: int = Field(..., description="Number of folds for cross-validation")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("x_train", "y_train", mode="before")
    def validate_ndarray(cls, value, info):
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{info.field_name} must be a numpy ndarray.")
        if value.size == 0:
            raise ValueError(f"{info.field_name} cannot be empty.")
        return value

    @field_validator("cv")
    def validate_cv(cls, value):
        if not isinstance(value, int) or value <= 1:
            raise ValueError("cv must be an integer greater than 1.")
        return value


class ModelEvaluationInput(BaseModel):
    test_data: np.ndarray
    predictions: np.ndarray
    residuals: Optional[np.ndarray] = Field(..., description="Residuals (optional)")

    # Set arbitrary_types_allowed to allow numpy.ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("predictions")
    def validate_array_lengths(cls, predictions, info: ValidationInfo):
        test_data = info.data.get("test_data")  # Access test_data via info.data
        if test_data is not None and len(test_data) != len(predictions):
            raise ValueError("Predictions must have the same length as test_data.")
        return predictions

    @field_validator("residuals")
    def validate_residuals(cls, residuals, info: ValidationInfo):
        test_data = info.data.get("test_data")  # Access test_data via info.data
        if residuals is not None and test_data is not None and len(residuals) != len(test_data):
            raise ValueError("Residuals must have the same length as test_data.")
        return residuals
