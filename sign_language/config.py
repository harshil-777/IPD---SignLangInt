"""Centralized configuration for the sign language interpreter.

Every tunable constant (paths, sequence length, thresholds, ports) lives
here so scripts and modules don't each hardcode their own copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FeatureConfig:
    num_landmarks: int = 21
    coords_per_landmark: int = 3
    joint_angle_groups: tuple[tuple[int, int, int], ...] = (
        (1, 2, 3), (2, 3, 4), (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12), (13, 14, 15),
        (14, 15, 16), (17, 18, 19), (18, 19, 20),
    )

    @property
    def features_per_hand(self) -> int:
        return self.num_landmarks * self.coords_per_landmark + len(self.joint_angle_groups)

    @property
    def features_per_frame(self) -> int:
        return self.features_per_hand * 2  # right hand + left hand


@dataclass(frozen=True)
class SequenceConfig:
    seq_len: int = 30
    no_hand_reset_limit: int = 10


@dataclass(frozen=True)
class ModelPaths:
    hand_landmarker_task: Path = PROJECT_ROOT / "hand_landmarker.task"
    gesture_model: Path = PROJECT_ROOT / "gesture_lstm.keras"
    label_encoder: Path = PROJECT_ROOT / "label_encoder.pkl"
    dataset_dir: Path = PROJECT_ROOT / "dynamic_dataset"


@dataclass(frozen=True)
class InferenceConfig:
    confidence_threshold: float = 0.50
    stable_window: int = 4
    predict_every_n_frames: int = 3
    smoothing_alpha: float = 0.5  # EMA weight given to the newest frame


@dataclass(frozen=True)
class ServerConfig:
    host: str = "localhost"
    port: int = 8000

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


FEATURES = FeatureConfig()
SEQUENCE = SequenceConfig()
PATHS = ModelPaths()
INFERENCE = InferenceConfig()
SERVER = ServerConfig()
