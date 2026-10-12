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
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    v1_u = v1 / n1
    v2_u = v2 / n2
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return float(np.degrees(angle))


def extract_hand_features(landmark_coords: np.ndarray, config: FeatureConfig = FEATURES) -> list[float]:
    """Convert (21, 3) hand-landmark coordinates into a wrist-centered,
    scale-normalized feature vector: flattened coordinates + joint angles.
    """
    if landmark_coords.shape[0] != config.num_landmarks or landmark_coords.shape[1] != config.coords_per_landmark:
        landmark_coords = landmarks_to_coords(landmark_coords)

    wrist = landmark_coords[0]
    shifted = landmark_coords - wrist
    max_dist = float(np.max(np.linalg.norm(shifted, axis=1)))
    normalized = shifted / max_dist if max_dist > 1e-6 else shifted
    flattened = normalized.flatten().tolist()

    angles = [
        calculate_angle(landmark_coords[a] - landmark_coords[b], landmark_coords[c] - landmark_coords[b])
        for a, b, c in config.joint_angle_groups
    ]
    return flattened + angles


def landmarks_to_coords(hand_landmarks) -> np.ndarray:
    """Convert MediaPipe hand landmarks or a raw (N, 3) coordinate array
    into a NumPy array of shape (21, 3). Always pads or trims to exactly 21
    landmarks so feature vector size (73) and joint angle indices are guaranteed.
    """
    if isinstance(hand_landmarks, np.ndarray):
        arr = np.asarray(hand_landmarks, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            coords = arr[:, :3]
        else:
            coords = np.zeros((FEATURES.num_landmarks, 3), dtype=np.float32)
    else:
        coords_list: list[list[float]] = []
        for lm in hand_landmarks:
            if hasattr(lm, "x") and hasattr(lm, "y") and hasattr(lm, "z"):
                coords_list.append([float(lm.x), float(lm.y), float(lm.z)])
            elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 3:
                coords_list.append([float(lm[0]), float(lm[1]), float(lm[2])])
            else:
                raise TypeError(f"Unsupported landmark entry: {type(lm)!r}")

        if coords_list:
            coords = np.array(coords_list, dtype=np.float32)
        else:
            coords = np.zeros((FEATURES.num_landmarks, 3), dtype=np.float32)

    # Ensure shape is exactly (21, 3)
    if coords.shape[0] < FEATURES.num_landmarks:
        pad_val = coords[-1:] if coords.shape[0] > 0 else np.zeros((1, 3), dtype=np.float32)
        pad = np.repeat(pad_val, FEATURES.num_landmarks - coords.shape[0], axis=0)
        coords = np.vstack([coords, pad])
    elif coords.shape[0] > FEATURES.num_landmarks:
        coords = coords[:FEATURES.num_landmarks]

    return coords
