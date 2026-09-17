"""Hand-landmark feature extraction.

This is the single source of truth for turning raw MediaPipe hand
landmarks into the normalized feature vector the model was trained on.
Previously this logic was copy-pasted across the data collector, the
augmentation script, and the live app; any change (e.g. a new joint
angle) had to be made in three places and could silently drift out of
sync. It now lives here once and every caller reuses it.
"""
from __future__ import annotations

import numpy as np

from .config import FEATURES, FeatureConfig


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return float(np.degrees(angle))


def extract_hand_features(landmark_coords: np.ndarray, config: FeatureConfig = FEATURES) -> list[float]:
    """Convert (21, 3) hand-landmark coordinates into a wrist-centered,
    scale-normalized feature vector: flattened coordinates + joint angles.
    """
    wrist = landmark_coords[0]
    shifted = landmark_coords - wrist
    max_dist = np.max(np.linalg.norm(shifted, axis=1))
    normalized = shifted / max_dist if max_dist > 0 else shifted
    flattened = normalized.flatten().tolist()

    angles = [
        calculate_angle(landmark_coords[a] - landmark_coords[b], landmark_coords[c] - landmark_coords[b])
        for a, b, c in config.joint_angle_groups
    ]
    return flattened + angles


def landmarks_to_coords(hand_landmarks) -> np.ndarray:
    """Convert a MediaPipe hand-landmark list into an (21, 3) array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
