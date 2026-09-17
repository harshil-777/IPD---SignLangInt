import os
import glob
import pandas as pd
import numpy as np
import uuid
import csv
from datetime import datetime

OUTPUT_DIR = "dynamic_dataset"
SEQ_LEN = 30

JOINT_GROUPS = [
    (1, 2, 3), (2, 3, 4), (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12), (13, 14, 15),
    (14, 15, 16), (17, 18, 19), (18, 19, 20)
]

def calculate_angle(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)

def process_hand_sequence(base_coords):
    """
    Given (21, 3) base coordinates for one hand:
    - If hand is absent (all zeros), returns (30, 73) of zeros.
    - If present, generates 30 frames with micro-tremor noise on 3D coordinates,
      re-centers on wrist, normalizes scale, and geometrically recalculates the 10 angles.
    """
    if np.all(np.isclose(base_coords, 0.0)):
        return np.zeros((SEQ_LEN, 73), dtype=np.float32)

    hand_features = []
    for _ in range(SEQ_LEN):
        # 1. Add realistic micro-tremor noise directly to 3D joint coordinates
        jitter = np.random.normal(0, 0.005, size=(21, 3))
        frame_coords = base_coords + jitter

        # 2. Re-center wrist at (0, 0, 0)
        wrist = frame_coords[0]
        shifted_coords = frame_coords - wrist

        # 3. Normalize maximum distance
        max_dist = np.max(np.linalg.norm(shifted_coords, axis=1))
        normalized_coords = shifted_coords / max_dist if max_dist > 0 else shifted_coords
        flattened_coords = normalized_coords.flatten().tolist()

        # 4. Geometrically recalculate angles based on the noisy coordinates
        angles = []
        for a, b, c in JOINT_GROUPS:
            v1 = frame_coords[a] - frame_coords[b]
            v2 = frame_coords[c] - frame_coords[b]
            angles.append(calculate_angle(v1, v2))

        hand_features.append(flattened_coords + angles)

    return np.array(hand_features, dtype=np.float32)

def augment_sequence(base_features):
    """
    Takes 146 static features (63 R_coords + 10 R_angles + 63 L_coords + 10 L_angles).
    Extracts 3D coordinates, applies micro-tremor, and recalculates angles geometrically.
    Returns (30, 146) sequence.
    """
    # Extract original 3D coordinates (first 63 values for each hand)
    right_coords = base_features[0:63].reshape(21, 3)
    left_coords = base_features[73:136].reshape(21, 3)

    right_seq = process_hand_sequence(right_coords) # (30, 73)
    left_seq = process_hand_sequence(left_coords)   # (30, 73)

    return np.concatenate([right_seq, left_seq], axis=1).astype(np.float32) # (30, 146)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    manifest_exists = os.path.exists(manifest_path)
    
    total_converted = 0
    
    with open(manifest_path, "a", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        if not manifest_exists:
            writer.writerow(["file_path", "label", "frames", "source_type", "created_at"])
            
        # Auto-discover all .csv files in the current working directory
        csv_files = [f for f in glob.glob("*.csv") if not f.startswith("manifest")]
        if not csv_files:
            print("No CSV files found in the current directory.")

        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                print(f"Skipping {csv_file}, not found.")
                continue
                
            print(f"Processing {csv_file}...")
            df = pd.read_csv(csv_file, header=None).dropna()
            
            # Features are all columns except the last one. Label is the last column.
            features = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values
            labels = df.iloc[:, -1].astype(str).values
            
            for i in range(len(features)):
                feat = features[i]
                label = labels[i].strip().upper()
                
                # Skip invalid rows
                if np.isnan(feat).any() or not label or label == "NAN":
                    continue
                
                label_dir = os.path.join(OUTPUT_DIR, label)
                os.makedirs(label_dir, exist_ok=True)
                
                seq = augment_sequence(feat)
                
                # Save as .npy
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                sample_name = f"{label}_staticconv_{timestamp}_{unique_id}.npy"
                sample_path = os.path.join(label_dir, sample_name)
                
                np.save(sample_path, seq)
                writer.writerow([sample_path, label, SEQ_LEN, "STATIC_CONVERTED", timestamp])
                
                total_converted += 1
                
                if total_converted % 1000 == 0:
                    print(f"  Converted {total_converted} samples...")

    print(f"\nDone! Converted a total of {total_converted} static frames into dynamic sequences.")

if __name__ == "__main__":
    main()
