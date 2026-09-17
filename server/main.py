"""FastAPI application entry point.

Run with: uvicorn server.main:app --reload
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from sign_language.config import SEQUENCE

from .gesture_predictor import GesturePredictorService
from .report_generator import GeminiReportGenerator, ReportGenerator, UnavailableReportGenerator
from .schemas import PredictRequest, PredictResponse, ReportRequest, ReportResponse

app = FastAPI(title="Sign Language Incident Report Backend")

predictor = GesturePredictorService()

_gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
report_generator: ReportGenerator = (
    GeminiReportGenerator(_gemini_api_key) if _gemini_api_key else UnavailableReportGenerator()
)


@app.post("/predict", response_model=PredictResponse)
def predict_gesture(req: PredictRequest) -> PredictResponse:
    if not predictor.is_ready:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(req.sequence) < SEQUENCE.seq_len:
        return PredictResponse(prediction="", confidence=0.0)

    prediction, confidence = predictor.predict(req.sequence)
    return PredictResponse(prediction=prediction, confidence=confidence)


@app.post("/generate_report", response_model=ReportResponse)
def generate_report(req: ReportRequest) -> ReportResponse:
    try:
        return ReportResponse(generated_sentence=report_generator.generate(req.compiled_inputs))
    except Exception as e:
        return ReportResponse(generated_sentence=f"Error generating Incident Report: {e}")
