import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

DATASET_DIR = "dynamic_dataset"
SEQ_LEN = 30
EXPECTED_FEATURES = 146
MANIFEST_COLUMNS = ["file_path", "label", "frames", "source_type", "created_at"]

def normalize_path(path):
    """Normalize file path to use consistent forward slashes for cross-platform matching."""
    return os.path.normpath(path).replace("\\", "/")

def validate_sequence_file(file_path):
    """
    Validates whether a .npy file matches the expected (30, 146) float sequence format.
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

def sync_manifest(dataset_dir=DATASET_DIR, verbose=True):
    """
    Scans the dataset directory on disk, reconciles it against manifest.csv:
    1. Detects new .npy files dropped into class folders and adds them to manifest.csv.
    2. Validates new files for correct shape (30, 146) and integrity.
    3. Prunes entries from manifest.csv whose files no longer exist on disk.
    4. Saves the updated, synchronized manifest.csv.
    """
    if not os.path.exists(dataset_dir):
        if verbose:
            print(f"[Sync] Dataset directory '{dataset_dir}' does not exist.")
        return {"added": 0, "removed": 0, "total": 0, "classes": 0}

    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    manifest_rows = []
    known_paths = set()

    initial_count = 0

    # Load existing manifest if present
    if os.path.exists(manifest_path):
        try:
            existing_df = pd.read_csv(manifest_path)
            initial_count = len(existing_df)
            # Ensure required columns exist
            for col in MANIFEST_COLUMNS:
                if col not in existing_df.columns:
                    if col == "source_type":
                        existing_df["source_type"] = "UNKNOWN"
                    else:
                        existing_df[col] = ""

            for _, row in existing_df.iterrows():
                fp = str(row["file_path"]).strip()
                norm_fp = normalize_path(fp)
                if os.path.exists(fp):
                    manifest_rows.append({
                        "file_path": fp,
                        "label": str(row["label"]).strip().upper(),
                        "frames": int(row["frames"]) if pd.notna(row["frames"]) else SEQ_LEN,
                        "source_type": str(row["source_type"]).strip() if pd.notna(row["source_type"]) else "UNKNOWN",
                        "created_at": str(row["created_at"]).strip() if pd.notna(row["created_at"]) else datetime.now().strftime("%Y%m%d_%H%M%S")
                    })
                    known_paths.add(norm_fp)
        except Exception as e:
            if verbose:
                print(f"[Sync] Warning: Could not parse existing manifest ({e}). A fresh manifest will be built.")
            manifest_rows = []
            known_paths = set()

    # Scan disk for all .npy files in subfolders
    pattern = os.path.join(dataset_dir, "*", "*.npy")
    disk_files = glob.glob(pattern)

    added_count = 0
    skipped_count = 0

    for file_path in disk_files:
        norm_fp = normalize_path(file_path)
        if norm_fp in known_paths:
            continue

        # Extract label from parent folder name: e.g. dynamic_dataset/HELP/abc.npy -> HELP
        label = os.path.basename(os.path.dirname(file_path)).strip().upper()

        # Validate sequence integrity
        is_valid, reason, _ = validate_sequence_file(file_path)
        if not is_valid:
            if verbose:
                print(f"[Sync] Skipping invalid file '{file_path}': {reason}")
            skipped_count += 1
            continue

        try:
            mtime = os.path.getmtime(file_path)
            created_at = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
        except Exception:
            created_at = datetime.now().strftime("%Y%m%d_%H%M%S")

        manifest_rows.append({
            "file_path": file_path,
            "label": label,
            "frames": SEQ_LEN,
            "source_type": "IMPORTED",
            "created_at": created_at
        })
        known_paths.add(norm_fp)
        added_count += 1

    # Reconstruct dataframe and save
    if manifest_rows:
        updated_df = pd.DataFrame(manifest_rows)[MANIFEST_COLUMNS]
        # Sort by label and file_path for clean readability
        updated_df = updated_df.sort_values(by=["label", "file_path"]).reset_index(drop=True)
    else:
        updated_df = pd.DataFrame(columns=MANIFEST_COLUMNS)

    updated_df.to_csv(manifest_path, index=False)

    removed_count = initial_count - (len(manifest_rows) - added_count)
    total_count = len(updated_df)
    unique_classes = updated_df["label"].nunique() if total_count > 0 else 0

    if verbose:
        print("=" * 60)
        print("  DATASET MANIFEST SYNCHRONIZATION REPORT")
        print("=" * 60)
        print(f"  Existing valid entries retained: {len(manifest_rows) - added_count}")
        print(f"  New external/disk files added:  {added_count}")
        if removed_count > 0:
            print(f"  Stale entries pruned (deleted):  {removed_count}")
        if skipped_count > 0:
            print(f"  Corrupted/invalid files skipped: {skipped_count}")
        print(f"  Total verified dataset samples:  {total_count}")
        print(f"  Total distinct sign classes:     {unique_classes}")
        print(f"  Manifest saved to:               {manifest_path}")
        print("=" * 60)

    return {
        "added": added_count,
        "removed": removed_count,
        "total": total_count,
        "classes": unique_classes
    }

def import_external_dataset(source_dir, target_dir=DATASET_DIR, move=False):
    """
    Imports .npy files from an external directory into target_dir.
    Can handle:
    1. Folder organized by labels: source_dir/<LABEL>/*.npy
    2. Flat folder with label prefixes: source_dir/<LABEL>_*.npy
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    print(f"\n[Import] Scanning '{source_dir}' for external sign language samples...")

    # Find all .npy files in source_dir recursively
    source_files = glob.glob(os.path.join(source_dir, "**", "*.npy"), recursive=True)
    if not source_files:
        print(f"[Import] No .npy files found in '{source_dir}'.")
        return

    imported_count = 0
    skipped_count = 0

    for src in source_files:
        # Determine label: check parent folder name first
        parent_dir_name = os.path.basename(os.path.dirname(src))
        if parent_dir_name and parent_dir_name != os.path.basename(source_dir):
            label = parent_dir_name.strip().upper()
        else:
            # Try to infer label from filename prefix: e.g. "HELP_123.npy" -> "HELP"
            fname = os.path.splitext(os.path.basename(src))[0]
            label = fname.split("_")[0].strip().upper()

        if not label:
            label = "UNKNOWN"

        # Validate sequence before copying
        is_valid, reason, _ = validate_sequence_file(src)
        if not is_valid:
            print(f"[Import] Skipping '{src}': {reason}")
            skipped_count += 1
            continue

        target_label_dir = os.path.join(target_dir, label)
        os.makedirs(target_label_dir, exist_ok=True)

        target_filename = os.path.basename(src)
        dst = os.path.join(target_label_dir, target_filename)

        # Prevent accidental overwrite by appending unique suffix if needed
        if os.path.exists(dst) and os.path.abspath(src) != os.path.abspath(dst):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            base, ext = os.path.splitext(target_filename)
            dst = os.path.join(target_label_dir, f"{base}_imported_{timestamp}{ext}")

        if os.path.abspath(src) != os.path.abspath(dst):
            if move:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)

        imported_count += 1

    print(f"[Import] Successfully placed {imported_count} files into '{target_dir}'. (Skipped: {skipped_count})")
    # Automatically synchronize manifest
    sync_manifest(target_dir, verbose=True)

def main():
    parser = argparse.ArgumentParser(description="Synchronize dataset with manifest.csv or import external datasets.")
    parser.add_argument("--import-from", type=str, default=None, help="Path to external directory containing .npy files to import.")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR, help="Target dynamic dataset directory (default: dynamic_dataset).")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying when importing from external path.")

    args = parser.parse_args()

    if args.import_from:
        import_external_dataset(args.import_from, target_dir=args.dataset_dir, move=args.move)
    else:
        sync_manifest(dataset_dir=args.dataset_dir, verbose=True)

if __name__ == "__main__":
    main()
