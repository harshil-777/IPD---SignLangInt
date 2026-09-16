import os
import pandas as pd
import numpy as np
import uuid
import csv
from datetime import datetime

# Files containing static frame data
STATIC_CSVS = [
    "dataset_features.csv", 
    "help.csv",
    "attack.csv",
    "house.csv",
    "male.csv",
    "numbers_dataset.csv"
]
OUTPUT_DIR = "dynamic_dataset"
SEQ_LEN = 30

def augment_sequence(base_features):
    """
    Takes a 1D array of 146 features and duplicates it 30 times to create a sequence.
    Adds realistic micro-tremor noise.
    Features: 0-62 (R_Coords), 63-72 (R_Angles), 73-135 (L_Coords), 136-145 (L_Angles)
    """
    seq = np.tile(base_features, (SEQ_LEN, 1))
    
    # Generate noise
    # Coordinates (range roughly -1 to 1): use std=0.005
    # Angles (range 0 to 180): use std=1.0 degree
    noise = np.zeros_like(seq)
    
    # Right hand
    noise[:, 0:63] = np.random.normal(0, 0.005, size=(SEQ_LEN, 63))
    noise[:, 63:73] = np.random.normal(0, 1.0, size=(SEQ_LEN, 10))
    # Left hand
    noise[:, 73:136] = np.random.normal(0, 0.005, size=(SEQ_LEN, 63))
    noise[:, 136:146] = np.random.normal(0, 1.0, size=(SEQ_LEN, 10))
    
    # Add noise but clip to reasonable bounds if necessary
    seq = seq + noise
    return seq.astype(np.float32)

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
            
        for csv_file in STATIC_CSVS:
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
