# Sign Language Incident Report Interpreter

A real-time sign language interpreter that recognizes dynamic hand gestures
(fingerspelled letters, digits, and whole-word signs) from a webcam feed and
walks a deaf complainant through a structured First Information Report (FIR)
questionnaire, then drafts a formal incident narrative with an LLM.

## Architecture

```
sign_language/     Core domain logic (framework-agnostic, reused everywhere)
  config.py           Centralized constants: paths, sequence length, thresholds
  features.py         Hand-landmark -> normalized feature vector (single source of truth)
  landmarker.py       MediaPipe HandLandmarker wrapper
  smoothing.py        EMA smoothing of per-frame features
  stabilizer.py       Turns noisy per-frame predictions into a stable label
  sentence.py         Builds words/sentences from stable predictions
  fir.py              FIR questionnaire state machine
  dataset.py          Saving/loading recorded gesture sequences
  augmentation.py     Synthetic sample generation for class balancing
  gesture_model.py    LSTM model: build, train, load, predict
  prediction_client.py  Client interface for talking to the backend

server/             FastAPI backend
  main.py             App wiring / routes
  schemas.py          Request/response models
  gesture_predictor.py  Loads the trained model, serves predictions
  report_generator.py   LLM report-drafting, behind a swappable interface

app/                Live webcam interpreter (the FastAPI backend must be running)
  live_interpreter.py  Orchestrates capture -> features -> prediction -> sentence -> FIR
  hud.py               On-screen overlay rendering
  main.py              Entry point

tools/              Offline CLIs for building the dataset/model
  collect_dataset.py    Record labeled gesture sequences from webcam
  balance_dataset.py    Synthesize samples to balance underrepresented classes
  train.py              Train the LSTM classifier
```

Each module has one job, and the parts that plausibly need to be swapped later
(the prediction transport, the report-generating LLM) sit behind small
interfaces (`PredictionClient`, `ReportGenerator`) instead of being called
directly, so a new implementation can be added without touching the callers.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY
```

`hand_landmarker.task`, `gesture_lstm.keras`, and `label_encoder.pkl` must be
present in the project root (already included).

## Running

```bash
# 1. Start the backend (loads the model once, serves predictions + reports)
uvicorn server.main:app --reload

# 2. In another terminal, start the live interpreter
python -m app.main
```

Controls: `N` next question, `B` previous question, `C` clear input,
`D`/Backspace delete last, `Enter` generate the incident report, `Esc` quit.

## Rebuilding the model

```bash
python -m tools.collect_dataset     # record more samples for a label
python -m tools.balance_dataset     # top up underrepresented classes
python -m tools.train               # retrain gesture_lstm.keras
```
