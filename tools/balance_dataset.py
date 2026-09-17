"""CLI: tops up every gesture class to a target sample count via
synthetic augmentation, so classes with fewer recordings don't bias
training.

Run with: python -m tools.balance_dataset
"""
from __future__ import annotations

from sign_language.augmentation import balance_dataset
from sign_language.dataset import SequenceDataset

TARGET_SAMPLES = 500


def main() -> None:
    dataset = SequenceDataset()
    print(f"Balancing all classes in '{dataset.root_dir}' to {TARGET_SAMPLES} samples...")
    balance_dataset(dataset, TARGET_SAMPLES)
    print(f"\nDataset augmentation and balancing complete! All classes have at least {TARGET_SAMPLES} samples.")


if __name__ == "__main__":
    main()
