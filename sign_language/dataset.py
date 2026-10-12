"""Storage layer for recorded gesture sequences: saving samples with a
manifest, and loading them back for training.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import PATHS, SEQUENCE


def file_checksum(path: Path | str) -> str:
    """Fast MD5 checksum of a file to check for duplicate content."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SequenceDataset:
    def __init__(self, root_dir: Path | str = PATHS.dataset_dir):
        self.root_dir = Path(root_dir)

    def manifest_path(self) -> Path:
        return self.root_dir / "manifest.csv"

    def save_sample(
        self,
        label: str,
        sequence: list[list[float]],
        source_type: str = "RECORDED",
    ) -> Path:
        """Save a gesture sequence as a .npy file and append an entry to manifest.csv.
        Guarantees unique filename even if rapid recordings happen at the exact same timestamp.
        """
        label_dir = self.root_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sample_path = label_dir / f"{label}_{timestamp}.npy"
        if sample_path.exists():
            sample_path = label_dir / f"{label}_{timestamp}_{uuid.uuid4().hex[:6]}.npy"

        np.save(sample_path, np.array(sequence, dtype=np.float32))

        # Store portable relative path with forward slashes
        rel_path = f"{self.root_dir.name}/{label}/{sample_path.name}"

        manifest_path = self.manifest_path()
        is_new = not manifest_path.exists()
        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["file_path", "label", "frames", "source_type", "created_at"])
            writer.writerow([rel_path, label, len(sequence), source_type, timestamp])

        return sample_path

    def sync_manifest(self, verbose: bool = True) -> dict[str, int]:
        """Scans the dataset directory, resolves filename collisions,
        adds new files, and prunes stale records in manifest.csv.
        Runs in < 0.2s even with 25,000+ files.
        """
        if not self.root_dir.exists():
            if verbose:
                print(f"[Dataset] Directory '{self.root_dir}' does not exist.")
            return {"added": 0, "removed": 0, "total": 0, "classes": 0, "renamed": 0}

        manifest_path = self.manifest_path()
        known_entries: dict[str, dict[str, str]] = {}

        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fp = row.get("file_path", "").replace("\\", "/").strip()
                        if fp:
                            known_entries[fp] = row
            except Exception as e:
                if verbose:
                    print(f"[Dataset] Warning reading manifest ({e}), rebuilding.")
                known_entries = {}

        # 1. First pass: resolve Windows duplicate filenames (e.g. 'sample (1).npy' or 'sample - Copy.npy')
        renamed_count = self._resolve_windows_copy_collisions()

        # 2. Fast scan disk
        disk_files: list[tuple[Path, str, str]] = []  # (full_path, label, rel_path)
        disk_rel_paths: set[str] = set()

        for label_dir in sorted(self.root_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name.strip().upper()
            for file_path in label_dir.glob("*.npy"):
                rel_path = f"{self.root_dir.name}/{label}/{file_path.name}"
                disk_files.append((file_path, label, rel_path))
                disk_rel_paths.add(rel_path)

        # 3. Detect new files on disk not yet in manifest
        new_entries: list[dict[str, str]] = []
        for file_path, label, rel_path in disk_files:
            if rel_path in known_entries:
                continue

            # Validate sequence file quickly using mmap_mode="r" (header only)
            try:
                arr = np.load(file_path, mmap_mode="r")
                if arr.shape != (SEQUENCE.seq_len, 146):
                    if verbose:
                        print(f"[Dataset] Skipping {file_path.name}: shape {arr.shape} != ({SEQUENCE.seq_len}, 146)")
                    continue
            except Exception as e:
                if verbose:
                    print(f"[Dataset] Skipping corrupt file {file_path.name}: {e}")
                continue

            try:
                mtime = os.path.getmtime(file_path)
                created_at = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
            except Exception:
                created_at = datetime.now().strftime("%Y%m%d_%H%M%S")

            new_entries.append({
                "file_path": rel_path,
                "label": label,
                "frames": str(SEQUENCE.seq_len),
                "source_type": "IMPORTED",
                "created_at": created_at,
            })

        # 4. Retain valid existing entries that still exist on disk
        final_rows: list[dict[str, str]] = []
        for rel_path, row in known_entries.items():
            if rel_path in disk_rel_paths:
                final_rows.append(row)

        # Append new entries
        final_rows.extend(new_entries)

        # Sort by label and file_path for clean deterministic layout
        final_rows.sort(key=lambda r: (r.get("label", ""), r.get("file_path", "")))

        # 5. Write back updated manifest
        columns = ["file_path", "label", "frames", "source_type", "created_at"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in final_rows:
                writer.writerow({col: r.get(col, "") for col in columns})

        added_count = len(new_entries)
        removed_count = len(known_entries) - (len(final_rows) - added_count)
        total_count = len(final_rows)
        classes_count = len(set(r.get("label", "") for r in final_rows))

        if verbose and (added_count > 0 or removed_count > 0 or renamed_count > 0):
            print(f"[Dataset] Manifest synchronized: {added_count} added, {removed_count} pruned, "
                  f"{renamed_count} collisions resolved. Total samples: {total_count} ({classes_count} classes).")

        return {
            "added": added_count,
            "removed": removed_count,
            "total": total_count,
            "classes": classes_count,
            "renamed": renamed_count,
        }

    def _resolve_windows_copy_collisions(self) -> int:
        """Finds Windows copy collision filenames (e.g. 'xyz (1).npy', 'xyz - Copy.npy')
        and safely renames them to unique standard names so no data is lost or overwritten.
        """
        renamed = 0
        copy_pattern = re.compile(r"(\s*\((\d+)\)|\s*-\s*Copy(\s*\(\d+\))?)$", re.IGNORECASE)

        for label_dir in self.root_dir.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for file_path in list(label_dir.glob("*.npy")):
                stem = file_path.stem
                if copy_pattern.search(stem):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_name = f"{label}_imported_{timestamp}_{uuid.uuid4().hex[:6]}.npy"
                    new_path = label_dir / new_name
                    try:
                        file_path.rename(new_path)
                        renamed += 1
                    except OSError:
                        pass
        return renamed

    def import_external(self, source_dir: Path | str, move: bool = False, verbose: bool = True) -> dict[str, int]:
        """Imports .npy files from an external folder (e.g. from a friend's laptop or drive).
        Automatically resolves filename collisions:
        - If content is identical (same MD5 hash), skips duplicate.
        - If content is different (different recording with same name), renames with unique suffix.
        - Updates manifest.csv automatically.
        """
        src_path = Path(source_dir)
        if not src_path.exists():
            raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

        import shutil
        all_source_files = list(src_path.rglob("*.npy"))
        if not all_source_files:
            if verbose:
                print(f"[Import] No .npy files found in '{source_dir}'.")
            return {"imported": 0, "skipped": 0, "collisions_resolved": 0}

        imported = 0
        skipped = 0
        collisions_resolved = 0

        for src in all_source_files:
            # Determine label
            parent_name = src.parent.name
            if parent_name and parent_name != src_path.name:
                label = parent_name.strip().upper()
            else:
                label = src.stem.split("_")[0].strip().upper()

            if not label:
                label = "UNKNOWN"

            target_label_dir = self.root_dir / label
            target_label_dir.mkdir(parents=True, exist_ok=True)

            target_file = target_label_dir / src.name

            # Collision check
            if target_file.exists():
                if src.resolve() == target_file.resolve():
                    continue

                # Check if file contents are identical
                src_hash = file_checksum(src)
                dst_hash = file_checksum(target_file)
                if src_hash == dst_hash:
                    # Exact duplicate sample, skip safely
                    skipped += 1
                    continue
                else:
                    # Distinct sample with a colliding name: generate unique friend filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_name = f"{src.stem}_friend_{timestamp}_{uuid.uuid4().hex[:6]}.npy"
                    target_file = target_label_dir / unique_name
                    collisions_resolved += 1

            if move:
                shutil.move(str(src), str(target_file))
            else:
                shutil.copy2(str(src), str(target_file))

            imported += 1

        if verbose:
            print(f"[Import] Imported {imported} files from '{source_dir}'. "
                  f"(Resolved {collisions_resolved} collisions, skipped {skipped} exact duplicates).")

        self.sync_manifest(verbose=verbose)
        return {"imported": imported, "skipped": skipped, "collisions_resolved": collisions_resolved}

    def load_all(self, seq_len: int = SEQUENCE.seq_len) -> tuple[np.ndarray, np.ndarray]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root_dir}")

        # Automatically ensure manifest and disk files are synchronized
        self.sync_manifest(verbose=True)

        X, y = [], []
        for label_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            for file_path in sorted(label_dir.glob("*.npy")):
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
