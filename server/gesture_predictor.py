"""Server-side wrapper that loads the trained gesture model once at
startup and exposes a safe `predict` for the API layer.
"""
from __future__ import annotations

from sign_language.config import FEATURES, SEQUENCE
from sign_language.gesture_model import GestureClassifier


class GesturePredictorService:
    def __init__(self):
        self._classifier: GestureClassifier | None = None
        try:
            self._classifier = GestureClassifier.load()
            self._classifier.warm_up(SEQUENCE.seq_len, FEATURES.features_per_frame)
            print("Gesture model loaded and warmed up.")
        except (OSError, IOError) as e:
            print(f"Warning: gesture model not found or failed to load ({e}). Predictions will fail.")

    @property
    def is_ready(self) -> bool:
        return self._classifier is not None

    def predict(self, sequence: list[list[float]]) -> tuple[str, float]:
        if self._classifier is None:
            raise RuntimeError("Gesture model not loaded")
        return self._classifier.predict(sequence)
