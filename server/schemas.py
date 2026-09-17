"""Request/response models for the backend API."""
from __future__ import annotations

from pydantic import BaseModel


class PredictRequest(BaseModel):
    sequence: list[list[float]]  # SEQUENCE.seq_len frames, each with FEATURES.features_per_frame values


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


class ReportRequest(BaseModel):
    compiled_inputs: str


class ReportResponse(BaseModel):
    generated_sentence: str
