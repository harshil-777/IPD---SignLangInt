"""CLI: trains the LSTM gesture classifier on the collected dataset.

Run with: python -m tools.train
"""
from __future__ import annotations

from sign_language.dataset import SequenceDataset
from sign_language.gesture_model import GestureClassifier


def main() -> None:
    print("Loading dynamic sequence dataset...")
    dataset = SequenceDataset()
    X, y = dataset.load_all()

    print(f"Loaded {len(X)} samples")
    print(f"Input shape per sample: {X.shape[1:]}")

    GestureClassifier.train(X, y)


if __name__ == "__main__":
    main()
