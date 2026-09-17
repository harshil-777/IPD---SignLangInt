"""
backfill_presence_flags.py
--------------------------
One-time migration: adds right_present and left_present flags to existing
147-column CSVs (73 right + 73 left + 1 label) --> 149-column CSVs
(73 right + 1 right_flag + 73 left + 1 left_flag + 1 label).

Presence is inferred by checking if the hand's feature block has a non-trivial
norm (threshold = 0.01). This is reliable because absent hands are stored as
all-zeros by the collector.

Usage:
    python backfill_presence_flags.py
"""

import os
import glob
import numpy as np
import pandas as pd

PRESENCE_THRESHOLD = 0.01  # Mean abs value below this → hand absent

def infer_presence(feature_block: np.ndarray) -> int:
    """Return 1 if hand appears present, 0 if absent (all-zeros block)."""
    return int(np.mean(np.abs(feature_block)) > PRESENCE_THRESHOLD)


def backfill_csv(input_path: str) -> str:
    """
    Reads an old 147-col CSV, inserts presence flags, writes a new file.
    Returns path to the new file.
    """
    df = pd.read_csv(input_path, header=None)

    if df.shape[1] == 149:
        print(f"  SKIP  {input_path} — already has 149 columns.")
        return input_path

    if df.shape[1] != 147:
        print(f"  WARN  {input_path} — unexpected column count ({df.shape[1]}), skipping.")
        return input_path

    right_block = df.iloc[:, 0:73].values.astype(float)    # cols 0-72
    left_block  = df.iloc[:, 73:146].values.astype(float)  # cols 73-145
    labels      = df.iloc[:, 146]                           # col 146

    right_flags = np.array([infer_presence(row) for row in right_block])
    left_flags  = np.array([infer_presence(row) for row in left_block])

    new_df = pd.DataFrame(
        np.hstack([
            right_block,
            right_flags.reshape(-1, 1),
            left_block,
            left_flags.reshape(-1, 1),
        ])
    )
    new_df[148] = labels.values   # append label as last column

    # Write to a new file (keep the original intact)
    base, ext = os.path.splitext(input_path)
    output_path = base + "_v2" + ext
    new_df.to_csv(output_path, index=False, header=False)

    right_present_count = int(right_flags.sum())
    left_present_count  = int(left_flags.sum())
    print(f"  OK    {input_path} -> {output_path}")
    print(f"        Rows: {len(df)} | Right present: {right_present_count} | Left present: {left_present_count}")
    return output_path


def main():
    csv_files = [f for f in glob.glob("*.csv") if not f.startswith("manifest")]
    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process:\n")
    for csv_file in csv_files:
        backfill_csv(csv_file)

    print("\nDone. New '_v2' files have been created alongside originals.")
    print("Once you verify the _v2 files, you can delete or rename the originals.")
    print("Remember: convert_static_to_dynamic.py now expects 149-column CSVs.")


if __name__ == "__main__":
    main()
