# core/ML/validators/validator_ModelTraining.py
from pydantic import BaseModel, field_validator, ValidationInfo, ConfigDict
import numpy as np
import pandas as pd
from typing import Union


class ModelTrainingInput(BaseModel):
    x_train: Union[np.ndarray, pd.DataFrame]
    y_train: Union[np.ndarray, pd.Series]
    x_test: Union[np.ndarray, pd.DataFrame]
    y_test: Union[np.ndarray, pd.Series]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("x_train", "y_train", "x_test", "y_test", mode="before")
    def validate_and_convert_arrays(cls, array, info: ValidationInfo):
        # Convert pandas objects to numpy arrays
        if isinstance(array, (pd.DataFrame, pd.Series)):
            return array.to_numpy()
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{info.field_name} must be a numpy array, DataFrame, or Series")
        if array.size == 0:
            raise ValueError(f"{info.field_name} cannot be empty")
        return array

    @field_validator("x_train", "x_test")
    def validate_feature_shapes(cls, array, info: ValidationInfo):
        if len(array.shape) != 2:
            raise ValueError(f"{info.field_name} must be a 2D array (features)")
        return array

    @field_validator("y_train", "y_test")
    def validate_target_shapes(cls, array, info: ValidationInfo):
        if len(array.shape) != 1:
            raise ValueError(f"{info.field_name} must be a 1D array (targets)")
        return array

    @field_validator("x_train", mode="after")
    def validate_consistent_samples(cls, x_train, info: ValidationInfo):
        y_train = info.data.get("y_train")
        if y_train is not None and x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of samples")
        return x_train

    @field_validator("x_test", mode="after")
    def validate_consistent_test_samples(cls, x_test, info: ValidationInfo):
        y_test = info.data.get("y_test")
        if y_test is not None and x_test.shape[0] != y_test.shape[0]:
            raise ValueError("x_test and y_test must have the same number of samples")
        return x_test
