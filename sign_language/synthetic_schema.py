"""Validation contract for synthetic landmark sequences."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class LandmarkSequenceContract(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    sequence: np.ndarray

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: np.ndarray) -> np.ndarray:
        if value.shape != (30, 146):
            raise ValueError("sequence must have shape (30, 146)")
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError("sequence must contain numeric values")
        if not np.isfinite(value).all():
            raise ValueError("sequence must not contain NaN or Inf values")
        return value