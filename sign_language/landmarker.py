"""MediaPipe hand-landmark detection, wrapped for reuse across the data
collector, the live app, and (indirectly) the augmentation pipeline.
"""
from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np

from .config import FEATURES, PATHS
from .features import extract_hand_features, landmarks_to_coords


class HandLandmarkerService:
    """Detects up to two hands in a frame and builds the per-frame
    gesture feature vector (right hand features, then left hand
    features) used everywhere else in the pipeline.
    """

    def __init__(self, model_path: Path = PATHS.hand_landmarker_task, num_hands: int = 2):
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=num_hands)
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def __enter__(self) -> "HandLandmarkerService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._landmarker.close()

    def detect(self, rgb_frame: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect(mp_image)

    def extract_frame_features(self, detection_result) -> list[float]:
        """Build a single frame's feature vector from a detection result:
        right hand features followed by left hand features, zero-filled
        for any hand that wasn't detected.

        MediaPipe handedness metadata can occasionally be missing, stale, or
        arrive in a different order from the landmark list. We therefore
        normalize the side label and fall back to the hand's x-position when
        necessary so the model still receives the expected right/left layout.
        """
        features_per_hand = FEATURES.features_per_hand
        right = [0.0] * features_per_hand
        left = [0.0] * features_per_hand

        hand_landmarks = getattr(detection_result, "hand_landmarks", None) or []
        if not hand_landmarks:
            return right + left

        handedness_data = getattr(detection_result, "handedness", None) or []

        def normalize_handedness(value: object | None) -> str | None:
            if value is None:
                return None
            if hasattr(value, "category_name"):
                value = value.category_name
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            value = str(value).strip().lower()
            return value if value in {"right", "left"} else None

        def infer_side_from_coords(coords: np.ndarray) -> str:
            avg_x = float(np.mean(coords[:, 0]))
            return "right" if avg_x >= 0.5 else "left"

        for idx, hand_landmarks_item in enumerate(hand_landmarks):
            coords = landmarks_to_coords(hand_landmarks_item)
            hand_features = extract_hand_features(coords)

            side = None
            if idx < len(handedness_data):
                candidate = handedness_data[idx]
                if isinstance(candidate, (list, tuple)) and candidate:
                    side = normalize_handedness(candidate[0])
                else:
                    side = normalize_handedness(candidate)
            if side is None:
                side = infer_side_from_coords(coords)

            if side == "right":
                right = hand_features
            elif side == "left":
                left = hand_features

        return right + left
