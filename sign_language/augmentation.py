"""Synthesizes additional training samples for underrepresented gesture
classes via small biomechanical/spatial/jitter perturbations, so class
imbalance doesn't bias the model toward whichever signs were recorded
more often.
"""
from __future__ import annotations

import random
import uuid

import numpy as np

from .config import FEATURES
from .dataset import SequenceDataset
from .features import extract_hand_features


def _perturb_hand_sequence(coords: np.ndarray) -> np.ndarray:
    """coords shape: (seq_len, 21, 3)."""
    posture_offset = np.random.normal(loc=0.0, scale=0.01, size=coords.shape[1:])
    coords = coords + posture_offset

    scale = np.random.uniform(0.95, 1.05)
    coords = coords * scale

    jitter = np.random.normal(loc=0.0, scale=0.002, size=coords.shape)
    return coords + jitter


def _recompute_hand_features(coords: np.ndarray) -> np.ndarray:
    return np.array([extract_hand_features(frame) for frame in coords], dtype=np.float32)


def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """sequence shape: (seq_len, features_per_frame), laid out as
    right-hand features followed by left-hand features.
    """
    hand_size = FEATURES.features_per_hand
    coords_size = FEATURES.num_landmarks * FEATURES.coords_per_landmark
    shape = (-1, FEATURES.num_landmarks, FEATURES.coords_per_landmark)

    right_coords = sequence[:, 0:coords_size].reshape(shape)
    left_coords = sequence[:, hand_size:hand_size + coords_size].reshape(shape)

    new_right = _recompute_hand_features(_perturb_hand_sequence(right_coords))
    new_left = _recompute_hand_features(_perturb_hand_sequence(left_coords))
    return np.concatenate([new_right, new_left], axis=1)


def balance_dataset(dataset: SequenceDataset, target_samples: int) -> None:
    """Tops up every class to at least `target_samples` by synthesizing
    augmented copies of existing samples.
    """
    for label in dataset.class_labels():
        files = dataset.class_files(label)
        count = len(files)
        if count == 0:
            print(f"Skipping {label} (0 files)")
            continue
        if count >= target_samples:
            continue

        needed = target_samples - count
        print(f"Class '{label}' has {count} files. Synthesizing {needed} new files...")
        for _ in range(needed):
            base_file = random.choice(files)
            seq = np.load(base_file)
            new_seq = augment_sequence(seq)
            new_path = base_file.parent / f"{label}_synth_{uuid.uuid4().hex[:8]}.npy"
            np.save(new_path, new_seq)
