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
        """
        features_per_hand = FEATURES.features_per_hand
        right = [0.0] * features_per_hand
        left = [0.0] * features_per_hand

        if detection_result.hand_landmarks and detection_result.handedness:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = detection_result.handedness[idx][0].category_name
                coords = landmarks_to_coords(hand_landmarks)
                hand_features = extract_hand_features(coords)

                if handedness == "Right":
                    right = hand_features
                elif handedness == "Left":
                    left = hand_features

        return right + left
