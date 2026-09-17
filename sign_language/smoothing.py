"""Exponential moving-average smoothing for per-frame feature vectors,
reducing camera/landmark jitter before frames are added to a sequence.
"""
from __future__ import annotations

import numpy as np


class FeatureSmoother:
    def __init__(self, alpha: float):
        self._alpha = alpha
        self._prev: np.ndarray | None = None

    @property
    def last(self) -> np.ndarray | None:
        return self._prev

    def reset(self) -> None:
        self._prev = None

    def smooth(self, features: np.ndarray) -> np.ndarray:
        if self._prev is not None:
            features = self._alpha * features + (1 - self._alpha) * self._prev
        self._prev = features
        return features
