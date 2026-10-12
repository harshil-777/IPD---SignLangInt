"""sync_dataset.py
---------------
Utility to synchronize dynamic_dataset with manifest.csv, resolve filename
collisions, and safely import dataset files from external sources (such as a
friend's laptop or drive).

Usage:
    # 1. Sync local dataset with manifest (resolves collisions, adds new files, prunes stale):
    python sync_dataset.py

    # 2. Safely import files from a friend's folder:
    python sync_dataset.py --import-from "path/to/friend_data"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from sign_language.config import PATHS, SEQUENCE
from sign_language.dataset import SequenceDataset

DATASET_DIR = str(PATHS.dataset_dir)
SEQ_LEN = SEQUENCE.seq_len
EXPECTED_FEATURES = 146
MANIFEST_COLUMNS = ["file_path", "label", "frames", "source_type", "created_at"]


def validate_sequence_file(file_path: str | Path):
    """Validates whether a .npy file matches the expected (30, 146) float sequence format.
    Returns (is_valid, reason, array_data).
    """
    try:
        data = np.load(file_path)
    except Exception as e:
        return False, f"Corrupted or unreadable .npy file ({e})", None

    if data.ndim != 2:
        return False, f"Invalid dimensions: expected 2D (time, features), got shape {data.shape}", None

    if data.shape[0] != SEQ_LEN:
        return False, f"Invalid sequence length: expected {SEQ_LEN} frames, got {data.shape[0]}", None

    if data.shape[1] != EXPECTED_FEATURES:
        return False, f"Invalid feature count: expected {EXPECTED_FEATURES} features, got {data.shape[1]}", None

    if np.isnan(data).any():
        return False, "Data contains NaN values", None

    return True, "Valid", data


def sync_manifest(dataset_dir: str | Path = DATASET_DIR, verbose: bool = True):
    """Scans the dataset directory on disk, reconciles it against manifest.csv:
    1. Resolves any Windows collision filenames ('sample (1).npy', 'sample - Copy.npy').
    2. Detects new .npy files and adds them to manifest.csv.
    3. Prunes entries from manifest.csv whose files no longer exist on disk.
    4. Saves the updated, synchronized manifest.csv.
    """
    dataset = SequenceDataset(root_dir=dataset_dir)
    res = dataset.sync_manifest(verbose=verbose)
    if verbose:
        print("=" * 60)
        print("  DATASET MANIFEST SYNCHRONIZATION REPORT")
        print("=" * 60)
        print(f"  New external/disk files added:    {res['added']}")
        print(f"  Filename collisions resolved:     {res['renamed']}")
        print(f"  Stale entries pruned (deleted):   {res['removed']}")
        print(f"  Total verified dataset samples:   {res['total']}")
        print(f"  Total distinct sign classes:      {res['classes']}")
        print(f"  Manifest saved to:                {dataset.manifest_path()}")
        print("=" * 60)
    return res


def import_external_dataset(source_dir: str | Path, target_dir: str | Path = DATASET_DIR, move: bool = False):
    """Imports .npy files from an external directory into target_dir.
    Automatically resolves collisions:
    - If identical content (same MD5 hash), avoids duplicate copies.
    - If distinct content with colliding name, renames with unique suffix.
    - Synchronizes manifest.csv.
    """
    dataset = SequenceDataset(root_dir=target_dir)
    return dataset.import_external(source_dir=source_dir, move=move, verbose=True)


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize dataset with manifest.csv or import external datasets from friends."
    )
    parser.add_argument(
        "--import-from",
        type=str,
        default=None,
        help="Path to external directory containing .npy files to import.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DATASET_DIR,
        help=f"Target dynamic dataset directory (default: {DATASET_DIR}).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying when importing from external path.",
    )

    args = parser.parse_args()

    if args.import_from:
        import_external_dataset(args.import_from, target_dir=args.dataset_dir, move=args.move)
    else:
        sync_manifest(dataset_dir=args.dataset_dir, verbose=True)


if __name__ == "__main__":
    main()
