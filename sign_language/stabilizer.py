"""Turns a noisy stream of per-frame predictions into a single stable
label, by requiring a prediction to dominate a rolling window before it's
accepted.
"""
from __future__ import annotations

from collections import Counter, deque


class PredictionStabilizer:
    def __init__(self, confidence_threshold: float, stable_window: int):
        self._confidence_threshold = confidence_threshold
        self._stable_window = stable_window
        self._buffer: deque[str] = deque(maxlen=stable_window)

    def reset(self) -> None:
        self._buffer.clear()

    def update(self, prediction: str, confidence: float) -> str:
        """Feed a new (prediction, confidence) pair. Returns the stable
        label once it dominates the recent window, otherwise "".
        """
        self._buffer.append(prediction if confidence >= self._confidence_threshold else "")

        candidates = [p for p in self._buffer if p]
        if not candidates:
            return ""

        label, count = Counter(candidates).most_common(1)[0]
        return label if count >= self._stable_window else ""
