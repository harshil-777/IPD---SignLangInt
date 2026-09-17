"""Storage layer for recorded gesture sequences: saving samples with a
manifest, and loading them back for training.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import PATHS, SEQUENCE


class SequenceDataset:
    def __init__(self, root_dir: Path | str = PATHS.dataset_dir):
        self.root_dir = Path(root_dir)

    def manifest_path(self) -> Path:
        return self.root_dir / "manifest.csv"

    def save_sample(self, label: str, sequence: list[list[float]]) -> Path:
        label_dir = self.root_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sample_path = label_dir / f"{label}_{timestamp}.npy"
        np.save(sample_path, np.array(sequence, dtype=np.float32))

        manifest_path = self.manifest_path()
        is_new = not manifest_path.exists()
        with open(manifest_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["file_path", "label", "frames", "created_at"])
            writer.writerow([str(sample_path), label, len(sequence), timestamp])

        return sample_path

    def load_all(self, seq_len: int = SEQUENCE.seq_len) -> tuple[np.ndarray, np.ndarray]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root_dir}")

        X, y = [], []
        for label_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            for file_path in label_dir.glob("*.npy"):
                seq = np.load(file_path)
                if seq.shape[0] != seq_len:
                    print(f"Skipping {file_path}: expected {seq_len} frames, got {seq.shape[0]}")
                    continue
                X.append(seq)
                y.append(label_dir.name)

        if not X:
            raise ValueError("No valid .npy sequence files found in the dataset.")

        return np.array(X, dtype=np.float32), np.array(y)

    def class_files(self, label: str) -> list[Path]:
        return list((self.root_dir / label).glob("*.npy"))

    def class_labels(self) -> list[str]:
        return [p.name for p in self.root_dir.iterdir() if p.is_dir()]
